import streamlit as st
import os
import fitz  # PyMuPDF
import tempfile
import time
import re
import io
import zipfile
from PIL import Image
import google.generativeai as genai
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------------------------------------
# 配置与初始化
# -------------------------------------------------

try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    HAS_API_KEY = True
except Exception:
    GEMINI_API_KEY = ""
    HAS_API_KEY = False

if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None

# -------------------------------------------------
# 核心功能函数
# -------------------------------------------------

def get_header_image(page):
    """截取页面顶部"""
    rect = page.rect
    clip_rect = fitz.Rect(0, 0, rect.width, rect.height * 0.25)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_code_by_rule(page):
    """规则提取：极速模式 (已修复 CUT 误判)"""
    try:
        # 扫描左上角
        target_rect = fitz.Rect(0, 0, 300, 150) 
        text_in_box = page.get_text("text", clip=target_rect)
        
        clean_text = text_in_box.upper().replace('\n', ' ').strip()
        
        # --- 核心修复：黑名单升级 ---
        # 这里的词绝对不会被当做机构代码
        BLACKLIST = [
            'THE', 'AND', 'RPT', 'ALL', 'USD', 'PDF', 'DAT', 'TIM', 'PAG', 'REC',
            'OUT', 'STA', 'FEE', 'REP', 'GRA', 'TOT', 'END', 'SUM', 'UNK', 'WHK',
            'ACC', 'NO.', 'NUM', 'BER', 'COU', 'UNT',
            'CUT', 'OFF', 'TRA', 'ACT', 'ION', 'DATE' # 新增：屏蔽 Transaction Cut-Off Date
        ]
        
        matches = re.findall(r'\b[A-Z]{3}\b', clean_text)
        valid_codes = [m for m in matches if m not in BLACKLIST]
        
        if len(valid_codes) > 0:
            # 优先返回第一个非黑名单代码
            return valid_codes[0]
        return None
    except Exception:
        return None

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def call_gemini_ai(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = """
    Analyze this document header.
    Find the 3-letter Agency Code (e.g., APO, FPL, OFS, IPP, WMG).
    It is usually in a box or at the top left.
    
    STRICTLY IGNORE: 
    - "Outstanding"
    - "Report"
    - "WHK" (Account No)
    - "Fee"
    - "Cut-Off" (Date)
    - "Transaction"
    
    Return JSON: {"code": "XXX"}
    """
    response = model.generate_content([prompt, image])
    return response.text

def extract_code_hybrid(page, api_key, page_num):
    # 1. 规则优先
    rule_code = extract_code_by_rule(page)
    if rule_code:
        return rule_code
    
    # 2. AI 兜底
    if not api_key:
        return "UNKNOWN"
        
    try:
        header_img = get_header_image(page)
        ai_response = call_gemini_ai(header_img, api_key)
        clean_json = ai_response.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        ai_code = data.get('code', 'UNKNOWN')
        
        # AI 结果二次过滤 (防止 AI 也读到 CUT)
        if ai_code in ['OUT', 'REP', 'FEE', 'WHK', 'UNK', 'CUT', 'OFF']:
            return "UNKNOWN"
        return ai_code
    except Exception:
        return "UNKNOWN"

def generate_filename(code, page_text):
    if "Outstanding" in page_text:
        return f"Rpt 614-{code} Outstanding.pdf"
    else:
        return f"Rpt 615-{code} MF.pdf"

def process_pdf(uploaded_file, progress_bar, status_text):
    temp_path = None
    try:
        st.session_state.generated_files = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        
        page_groups = []
        current_group = []
        last_code = None
        
        # --- 扫描阶段 ---
        for i in range(total_pages):
            page = doc[i]
            page_text = page.get_text()
            
            progress_bar.progress((i + 1) / total_pages)
            status_text.text(f"正在分析第 {i+1}/{total_pages} 页...")
            
            if "End of Report" in page_text or "Grand Total" in page_text:
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                    current_group = []
                    last_code = None
                continue

            code = extract_code_hybrid(page, GEMINI_API_KEY, i)
            
            # 逻辑修正
            if code == "UNKNOWN" and last_code:
                code = last_code
            if code == "UNKNOWN" and last_code is None:
                code = "Unclassified"

            if code != last_code:
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                current_group = [i]
                last_code = code
            else:
                current_group.append(i)
        
        if current_group:
            final_code = last_code if last_code else "Unclassified"
            page_groups.append({'code': final_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
            
        doc.close()
        
        # --- 生成阶段 ---
        if not page_groups:
            page_groups.append({'code': "ALL", 'pages': list(range(total_pages)), 'text': ""})

        status_text.text("正在打包文件...")
        source_doc = fitz.open(temp_path)
        
        for group in page_groups:
            code = group['code']
            pages = group['pages']
            if not pages: continue

            out_doc = fitz.open()
            for p in pages:
                out_doc.insert_pdf(source_doc, from_page=p, to_page=p)
            
            out_buffer = io.BytesIO()
            out_doc.save(out_buffer)
            out_doc.close()
            
            filename = generate_filename(code, group['text'])
            
            st.session_state.generated_files.append({
                'filename': filename,
                'content': out_buffer.getvalue(),
                'code': code,
                'page_count': len(pages),
                'page_range': f"{min(pages)+1}-{max(pages)+1}"
            })
            
        source_doc.close()
        st.session_state.processing_complete = True
        
        if st.session_state.generated_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in st.session_state.generated_files:
                    zf.writestr(f['filename'], f['content'])
            zip_buffer.seek(0)
            st.session_state.zip_data = zip_buffer
            
        return st.session_state.generated_files

    except Exception as e:
        st.error(f"处理出错: {str(e)}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------
# UI 界面
# -------------------------------------------------

st.set_page_config(page_title="PDF 报表拆分系统", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 PDF 报表自动拆分系统")
st.markdown("上传包含多个机构的 PDF 报表，系统将自动识别机构代码并拆分为独立文件。")

with st.sidebar:
    st.header("系统状态")
    if HAS_API_KEY:
        st.success("✅ AI 引擎已就绪")
    else:
        st.info("ℹ️ 极速规则模式")
    
    st.divider()
    st.markdown("**使用说明**")
    st.markdown("1. 拖拽 PDF 上传")
    st.markdown("2. 点击开始拆分")
    st.markdown("3. 下载结果")

uploaded_file = st.file_uploader("📂 上传 PDF 文件", type="pdf")

if uploaded_file:
    if st.button("🚀 开始拆分", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        
        files = process_pdf(uploaded_file, progress, status)
        
        progress.progress(100)
        status.text("✅ 处理完成")
        
        if not files:
            st.error("未生成文件，请检查 PDF 内容。")

if st.session_state.processing_complete and st.session_state.generated_files:
    st.divider()
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"🎉 拆分结果 ({len(st.session_state.generated_files)} 个文件)")
    with c2:
        if st.session_state.zip_data:
            st.download_button(
                label="📦 下载全部 (ZIP)",
                data=st.session_state.zip_data,
                file_name="split_reports.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
    
    st.write("")

    for i, f in enumerate(st.session_state.generated_files):
        with st.container():
            col_info, col_prev, col_dl = st.columns([6, 2, 2])
            
            with col_info:
                if f['code'] == "Unclassified":
                    st.warning(f"⚠️ **{f['filename']}** (未识别代码)")
                else:
                    st.markdown(f"### 📄 {f['filename']}")
                
                st.caption(f"🏷️ 机构: **{f['code']}**  |  📑 页数: **{f['page_count']}**  |  📍 范围: p{f['page_range']}")
            
            with col_prev:
                if st.button("👁️ 预览", key=f"p_{i}"):
                    try:
                        with fitz.open(stream=f['content'], filetype="pdf") as doc:
                            st.image(doc[0].get_pixmap().tobytes("png"), caption="首页预览", use_container_width=True)
                    except:
                        st.error("无法预览")
            
            with col_dl:
                st.download_button(
                    "⬇️ 下载",
                    data=f['content'],
                    file_name=f['filename'],
                    mime="application/pdf",
                    key=f"d_{i}",
                    use_container_width=True
                )
            st.divider()

