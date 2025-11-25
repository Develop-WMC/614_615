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
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []

# -------------------------------------------------
# 核心功能函数
# -------------------------------------------------

def get_header_image(page):
    """截取页面顶部，用于 AI 分析"""
    rect = page.rect
    # 截取顶部 25%
    clip_rect = fitz.Rect(0, 0, rect.width, rect.height * 0.25)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_code_by_rule(page):
    """
    规则提取：扩大范围，严格黑名单
    """
    try:
        # 1. 扩大扫描范围：左上角 300x150，防止代码因为页边距偏移而漏掉
        target_rect = fitz.Rect(0, 0, 300, 150) 
        text_in_box = page.get_text("text", clip=target_rect)
        
        clean_text = text_in_box.upper().replace('\n', ' ').strip()
        
        # 2. 黑名单：这些词绝对不是机构代码
        BLACKLIST = [
            'THE', 'AND', 'RPT', 'ALL', 'USD', 'PDF', 'DAT', 'TIM', 'PAG', 'REC',
            'OUT', 'STA', 'FEE', 'REP', 'GRA', 'TOT', 'END', 'SUM', 'UNK', 'WHK',
            'ACC', 'NO.', 'NUM', 'BER', 'COU', 'UNT'
        ]
        
        # 提取所有3字母单词
        matches = re.findall(r'\b[A-Z]{3}\b', clean_text)
        valid_codes = [m for m in matches if m not in BLACKLIST]
        
        # 调试日志
        # st.session_state.debug_logs.append(f"Rule found: {valid_codes}")
        
        if len(valid_codes) > 0:
            # 优先返回第一个看起来像代码的
            return valid_codes[0]
            
        return None
    except Exception:
        return None

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def call_gemini_ai(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    Analyze this document header.
    Find the 3-letter Agency Code (e.g., APO, FPL, OFS).
    It is usually in a box or at the top left.
    
    IGNORE: "Outstanding", "Report", "WHK" (if account number), "Fee".
    
    Return JSON: {"code": "XXX"}
    """
    
    response = model.generate_content([prompt, image])
    return response.text

def extract_code_hybrid(page, api_key, page_num, status_text):
    # 1. 规则优先
    rule_code = extract_code_by_rule(page)
    if rule_code:
        return rule_code
    
    # 2. AI 兜底
    if not api_key:
        return "UNKNOWN"
        
    status_text.text(f"第 {page_num+1} 頁: 正在 AI 分析...")
    try:
        header_img = get_header_image(page)
        ai_response = call_gemini_ai(header_img, api_key)
        clean_json = ai_response.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        ai_code = data.get('code', 'UNKNOWN')
        
        if ai_code in ['OUT', 'REP', 'FEE', 'WHK', 'UNK']:
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
        # 重置
        st.session_state.generated_files = []
        st.session_state.debug_logs = []
        
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
            
            # 摘要页处理
            if "End of Report" in page_text or "Grand Total" in page_text:
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                    current_group = []
                    last_code = None
                continue

            # 提取代码
            code = extract_code_hybrid(page, GEMINI_API_KEY, i, status_text)
            
            # 核心修复：如果代码是 UNKNOWN，但上一页有代码，则沿用上一页
            if code == "UNKNOWN" and last_code:
                code = last_code
            
            # 核心修复：如果第一页就是 UNKNOWN，强制标记为 Unclassified，防止被丢弃
            if code == "UNKNOWN" and last_code is None:
                code = "Unclassified"

            # 分组逻辑
            if code != last_code:
                if current_group:
                    # 结束上一组
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                # 开始新组
                current_group = [i]
                last_code = code
            else:
                # 同一组
                current_group.append(i)
        
        # 处理最后一组
        if current_group:
            # 即使 last_code 是 None (理论上上面处理了，这里防万一)，也保存
            final_code = last_code if last_code else "Unclassified"
            page_groups.append({'code': final_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
            
        doc.close()
        
        # --- 生成文件阶段 ---
        if not page_groups:
            st.error("警告：未能识别任何页面分组。将尝试导出整个文件。")
            # 兜底：如果分组为空，把所有页面当做一个文件
            page_groups.append({'code': "ALL", 'pages': list(range(total_pages)), 'text': ""})

        status_text.text(f"正在生成 {len(page_groups)} 个文件...")
        
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
        st.error(f"Critical Error: {str(e)}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------
# UI 界面
# -------------------------------------------------

st.set_page_config(page_title="PDF 报表拆分系统", layout="wide")

st.title("📊 PDF 报表拆分系统 (完整版)")
st.markdown("""
**功能说明**：
1. **自动拆分**：根据左上角机构代码 (APO, FPL 等) 拆分报表。
2. **智能纠错**：自动忽略 "Outstanding", "WHK" 等干扰词。
3. **兜底保证**：即使识别失败，也会生成 "Unclassified" 文件，绝不丢失页面。
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统设置")
    user_api_key = st.text_input("Gemini API Key (可选)", value=GEMINI_API_KEY, type="password", help="输入 Key 可提高识别准确率，不输入则使用规则模式")
    if user_api_key: GEMINI_API_KEY = user_api_key
    
    st.divider()
    st.info("提示：如果结果中出现 'Unclassified' 文件，说明该部分页面无法通过规则识别代码，建议配置 API Key 重试。")

uploaded_file = st.file_uploader("📂 请上传 PDF 报表文件", type="pdf")

if uploaded_file:
    st.write(f"已加载文件: `{uploaded_file.name}`")
    
    if st.button("🚀 开始拆分处理", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        files = process_pdf(uploaded_file, progress, status)
        
        progress.progress(100)
        status.text("✅ 处理完成！")
        
        if not files:
            st.error("错误：未生成任何文件。请检查 PDF 是否加密或为空。")

# 结果展示区域
if st.session_state.processing_complete and st.session_state.generated_files:
    st.divider()
    
    # 顶部统计与下载
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"🎉 处理结果: 共 {len(st.session_state.generated_files)} 个文件")
    with c2:
        if st.session_state.zip_data:
            st.download_button(
                label="📦 一键下载所有文件 (ZIP)",
                data=st.session_state.zip_data,
                file_name="split_reports.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
    
    st.write("") # Spacer

    # 详细文件列表 (恢复完整 UI)
    for i, f in enumerate(st.session_state.generated_files):
        # 给每个文件一个卡片样式
        with st.container():
            # 使用列布局：图标+信息 | 预览 | 下载
            col_info, col_prev, col_dl = st.columns([5, 2, 2])
            
            with col_info:
                # 判断是否为未分类，给不同颜色
                if f['code'] == "Unclassified":
                    st.warning(f"⚠️ **{f['filename']}**")
                    st.caption("未能识别机构代码，请手动检查内容")
                else:
                    st.markdown(f"📄 **{f['filename']}**")
                
                st.caption(f"🏷️ 机构代码: `{f['code']}` | 📄 页数: `{f['page_count']}` | 📑 范围: `p{f['page_range']}`")
            
            with col_prev:
                # 预览按钮
                if st.button("👁️ 预览首页", key=f"prev_{i}", use_container_width=True):
                    try:
                        with fitz.open(stream=f['content'], filetype="pdf") as doc:
                            page = doc[0]
                            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                            st.image(pix.tobytes("png"), use_container_width=True)
                    except:
                        st.error("预览失败")
            
            with col_dl:
                st.download_button(
                    "⬇️ 下载 PDF",
                    data=f['content'],
                    file_name=f['filename'],
                    mime="application/pdf",
                    key=f"dl_{i}",
                    use_container_width=True
                )
            
            st.markdown("---")
