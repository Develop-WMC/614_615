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
except Exception:
    GEMINI_API_KEY = ""

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
    """只截取页面顶部极小区域传给 AI"""
    rect = page.rect
    # 只取顶部 20% (进一步缩小范围，防止看到太多干扰)
    clip_rect = fitz.Rect(0, 0, rect.width, rect.height * 0.2)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip_rect)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_code_by_rule(page):
    """
    规则提取 (修正版)
    """
    try:
        # 1. 缩小坐标范围！只看极左上角
        # 之前的 (250, 150) 太大了，扫到了标题里的 Outstanding
        # 现在改为 (10, 10, 120, 80)，只盯着那个小方框
        target_rect = fitz.Rect(10, 10, 120, 100) 
        text_in_box = page.get_text("text", clip=target_rect)
        
        clean_text = text_in_box.upper().replace('\n', ' ').strip()
        
        # 2. 严格的黑名单
        # 这里的词绝对不能作为机构代码返回
        BLACKLIST = [
            'THE', 'AND', 'RPT', 'ALL', 'USD', 'PDF', 'DAT', 'TIM', 'PAG', 'REC',
            'OUT', 'STA', 'FEE', 'REP', 'GRA', 'TOT', 'END', 'SUM', 'UNK', 'WHK'
        ]
        
        matches = re.findall(r'\b[A-Z]{3}\b', clean_text)
        valid_codes = [m for m in matches if m not in BLACKLIST]
        
        # 必须非常确信才返回
        if len(valid_codes) == 1:
            return valid_codes[0]
            
        return None
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini_ai(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    Analyze this document header image.
    Task: Identify the 3-letter Agency Code inside the box at the top-left.
    
    STRICT RULES:
    1. IGNORE the word "Outstanding".
    2. IGNORE the word "Report".
    3. IGNORE "WHK" if it is part of an Account Number.
    4. The code is usually: APO, FPL, OFS, WMG, WCL, etc.
    
    Return ONLY the code in JSON format: {"code": "XXX"}
    """
    
    response = model.generate_content([prompt, image])
    return response.text

def extract_code_hybrid(page, api_key, page_num, status_text):
    # 1. 先试规则
    rule_code = extract_code_by_rule(page)
    if rule_code:
        return rule_code
    
    # 2. 再试 AI
    if not api_key:
        return "UNKNOWN"
        
    status_text.text(f"第 {page_num+1} 頁: 正在 AI 分析...")
    try:
        header_img = get_header_image(page)
        ai_response = call_gemini_ai(header_img, api_key)
        clean_json = ai_response.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        ai_code = data.get('code', 'UNKNOWN')
        
        # AI 结果二次过滤
        if ai_code in ['OUT', 'REP', 'FEE', 'WHK']:
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        
        page_groups = []
        current_group = []
        last_code = None
        
        # 扫描阶段
        for i in range(total_pages):
            page = doc[i]
            page_text = page.get_text()
            
            progress_bar.progress((i + 1) / total_pages)
            
            # 跳过摘要页
            if "End of Report" in page_text or "Grand Total" in page_text:
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                    current_group = []
                    last_code = None
                continue

            code = extract_code_hybrid(page, GEMINI_API_KEY, i, status_text)
            
            # 连续性修正：如果识别失败或识别出 OUT/WHK，沿用上一个
            if (code == "UNKNOWN" or code == "OUT") and last_code:
                code = last_code

            # 分组逻辑
            if code != last_code and code != "UNKNOWN":
                if current_group:
                    # 保存上一组
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                # 开启新组
                current_group = [i]
                last_code = code
            elif last_code is not None:
                # 同一组
                current_group.append(i)
            elif code != "UNKNOWN":
                # 第一页
                current_group = [i]
                last_code = code
        
        # 最后一组
        if current_group and last_code:
            page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
            
        doc.close()
        
        # 生成文件阶段
        status_text.text("正在拆分并生成文件...")
        st.session_state.generated_files = []
        source_doc = fitz.open(temp_path)
        
        for group in page_groups:
            code = group['code']
            # 双重保险：如果代码还是 OUT，强制改为 UNKNOWN
            if code == "OUT": code = "UNKNOWN"
            
            pages = group['pages']
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
        
        # 生成 ZIP
        if st.session_state.generated_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in st.session_state.generated_files:
                    zf.writestr(f['filename'], f['content'])
            zip_buffer.seek(0)
            st.session_state.zip_data = zip_buffer
            
        return st.session_state.generated_files

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------
# UI 界面
# -------------------------------------------------

st.set_page_config(page_title="PDF 智能拆分", layout="wide")

st.title("🚀 PDF 报表拆分 (修正版)")
st.markdown("已修复 'OUT' 误判问题，确保正确按机构代码拆分。")

# 侧边栏
with st.sidebar:
    st.header("API 设置")
    user_api_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    if user_api_key: GEMINI_API_KEY = user_api_key

uploaded_file = st.file_uploader("上传 PDF", type="pdf")

if uploaded_file:
    if st.button("开始拆分", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        files = process_pdf(uploaded_file, progress, status)
        progress.progress(100)
        status.text("完成！")
        if files:
            st.success(f"成功拆分出 {len(files)} 个文件")

# 结果展示 - 恢复详细列表样式
if st.session_state.processing_complete and st.session_state.generated_files:
    st.divider()
    
    # 顶部下载 ZIP
    if st.session_state.zip_data:
        st.download_button(
            label="📦 下载全部文件 (ZIP)",
            data=st.session_state.zip_data,
            file_name="split_reports.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
    
    st.write("---")
    st.subheader("文件列表")
    
    # 使用更清晰的卡片式布局
    for i, f in enumerate(st.session_state.generated_files):
        with st.container():
            col1, col2, col3 = st.columns([5, 2, 2])
            with col1:
                st.markdown(f"### 📄 {f['filename']}")
                st.caption(f"机构代码: **{f['code']}** | 页数: {f['page_count']} (第 {f['page_range']} 页)")
            with col2:
                # 预览功能
                if st.button(f"预览首页", key=f"prev_{i}"):
                    with fitz.open(stream=f['content'], filetype="pdf") as doc:
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                        st.image(pix.tobytes("png"), caption=f"{f['filename']} - Page 1")
            with col3:
                st.download_button(
                    "⬇️ 下载 PDF",
                    data=f['content'],
                    file_name=f['filename'],
                    mime="application/pdf",
                    key=f"dl_{i}",
                    use_container_width=True
                )
            st.divider()
