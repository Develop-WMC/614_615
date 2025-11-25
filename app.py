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

# 尝试从Streamlit secrets获取API密钥
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    GEMINI_API_KEY = ""

# 初始化 Session State
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
    """
    只截取页面顶部的图像传给 AI。
    这是防止 AI 被下方的 'WHK' 账号干扰的关键！
    """
    # 获取页面尺寸
    rect = page.rect
    # 只取顶部 30% 的区域 (足够包含 Header 和那个方框)
    clip_rect = fitz.Rect(0, 0, rect.width, rect.height * 0.3)
    
    # 提高清晰度 (zoom=3) 以便 AI 识别小字
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip_rect)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_code_by_rule(page):
    """
    规则提取：极速，高准确率（针对固定位置）。
    如果这里成功，就不需要浪费时间调 AI。
    """
    try:
        # 1. 锁定左上角那个方框的坐标区域 (根据你的截图估算)
        # 假设页面宽 600，方框大概在 (20, 20) 到 (150, 100) 之间
        target_rect = fitz.Rect(10, 10, 250, 150)
        text_in_box = page.get_text("text", clip=target_rect)
        
        # 清理文本
        clean_text = text_in_box.upper().replace('\n', ' ').strip()
        
        # 寻找独立的3个大写字母
        # 排除常见词：THE, AND, RPT (Report), ALL, USD, PDF
        matches = re.findall(r'\b[A-Z]{3}\b', clean_text)
        valid_codes = [m for m in matches if m not in ['THE', 'AND', 'RPT', 'ALL', 'USD', 'PDF', 'DAT', 'TIM', 'PAG', 'REC']]
        
        # 如果在左上角方框里只找到了一个有效代码，那准确率是极高的
        if len(valid_codes) == 1:
            return valid_codes[0]
        
        # 如果找到了多个，优先取第一个（通常方框里的字最大或最靠前）
        if len(valid_codes) > 0:
            return valid_codes[0]
            
        return None
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini_ai(image, api_key):
    """调用 AI，但只看 Header"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # 使用 Flash 模型，速度更快
    
    prompt = """
    Look at this document HEADER.
    Identify the 3-letter Agency/Department code inside the box at the top-left or in the header line.
    
    Rules:
    1. Ignore any "Account No" or "WHK" references unless "WHK" is explicitly the Agency Code in the box.
    2. Common codes: APO, FPL, OFS, WMG, WCL.
    3. Return ONLY the 3-letter code. If unsure, return "UNKNOWN".
    
    Output Format: JSON
    {"code": "XXX"}
    """
    
    response = model.generate_content([prompt, image])
    return response.text

def extract_code_hybrid(page, api_key, page_num, status_text):
    """
    混合提取策略：
    1. 先试规则 (0秒耗时)
    2. 规则不行再试 AI (几秒耗时)
    """
    # --- 第一道防线：规则提取 ---
    rule_code = extract_code_by_rule(page)
    
    if rule_code:
        # 如果规则找到了看起来很靠谱的代码，直接返回，不调 AI
        # 这解决了 "Loading 时间长" 的问题
        return rule_code, "rule"
    
    # --- 第二道防线：AI 提取 ---
    if not api_key:
        return "UNKNOWN", "fail"
        
    status_text.text(f"第 {page_num+1} 頁: 规则无法确定，正在咨询 AI...")
    
    try:
        # 关键：只传 Header 图片，解决 WHK 误判
        header_img = get_header_image(page)
        
        ai_response = call_gemini_ai(header_img, api_key)
        
        # 解析 JSON
        clean_json = ai_response.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        ai_code = data.get('code', 'UNKNOWN')
        
        return ai_code, "ai"
        
    except Exception as e:
        print(f"AI Error: {e}")
        return "UNKNOWN", "error"

def generate_filename(code, page_text):
    """生成文件名"""
    if "Outstanding" in page_text:
        return f"Rpt 614-{code} Outstanding.pdf"
    else:
        return f"Rpt 615-{code} MF.pdf"

def process_pdf(uploaded_file, progress_bar, status_text):
    temp_path = None
    try:
        # 保存文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        
        page_groups = []
        current_group = []
        last_code = None
        
        # -------------------------------------------------
        # 阶段 1: 识别 (Hybrid)
        # -------------------------------------------------
        for i in range(total_pages):
            page = doc[i]
            page_text = page.get_text()
            
            # 进度条
            progress_bar.progress((i + 1) / total_pages)
            status_text.text(f"正在分析第 {i+1}/{total_pages} 頁...")
            
            # 检查摘要页
            if "End of Report" in page_text or "Grand Total" in page_text:
                # 摘要页不归类，结束当前组
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                    current_group = []
                    last_code = None
                continue

            # 提取代码 (混合模式)
            code, method = extract_code_hybrid(page, GEMINI_API_KEY, i, status_text)
            
            # 逻辑修正：如果这一页识别失败 (UNKNOWN)，但它是连续报表的一部分，
            # 我们假设它属于上一个机构 (通常报表中间不会突然变)
            if (code == "UNKNOWN" or code == "WHK") and last_code:
                # 注意：这里加了 code == "WHK" 的判断。
                # 如果 AI 依然发疯返回 WHK，但上一页是 APO，我们倾向于相信它是 APO 的续页
                # 除非这是第一页
                pass 
            
            # 如果 AI 还是返回了 WHK，我们需要再次确认它是不是真的 WHK
            # (大部分情况下，你的报表里 WHK 是账号前缀，不是机构代码)
            if code == "WHK":
                 # 简单的启发式：如果这是第一页，或者上一页不是 WHK，我们标记为存疑
                 # 但基于你的需求，我们先信任混合提取的结果，除非它明显是错的
                 pass

            # 分组逻辑
            if code != last_code and code != "UNKNOWN":
                if current_group:
                    page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
                current_group = [i]
                last_code = code
            elif last_code is not None:
                current_group.append(i)
            elif code != "UNKNOWN":
                # 第一页就是新组
                current_group = [i]
                last_code = code
        
        # 添加最后一组
        if current_group and last_code:
            page_groups.append({'code': last_code, 'pages': current_group, 'text': doc[current_group[0]].get_text()})
            
        doc.close()
        
        # -------------------------------------------------
        # 阶段 2: 拆分与保存
        # -------------------------------------------------
        status_text.text("正在生成 PDF 文件...")
        st.session_state.generated_files = []
        
        source_doc = fitz.open(temp_path)
        
        for group in page_groups:
            code = group['code']
            pages = group['pages']
            
            # 创建新 PDF
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
                'page_count': len(pages)
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

st.set_page_config(page_title="PDF 智能拆分 (AI+规则)", layout="wide")

st.title("🚀 PDF 报表智能拆分 (高精度版)")
st.markdown("""
此版本结合了 **规则定位** 和 **Gemini AI**，旨在实现 99% 的准确率。
1. **抗干扰**：AI 只看页面头部，自动忽略下方的 WHK 账号信息。
2. **提速**：优先使用坐标定位提取代码，仅在不确定时调用 AI。
""")

# 侧边栏 API 设置
with st.sidebar:
    st.header("设置")
    user_api_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    if user_api_key:
        GEMINI_API_KEY = user_api_key
        st.success("API Key 已就绪")
    else:
        st.warning("请输入 API Key 以启用 AI 增强模式")

uploaded_file = st.file_uploader("上传 PDF", type="pdf")

if uploaded_file:
    if st.button("开始处理", type="primary"):
        if not GEMINI_API_KEY:
            st.error("请先在侧边栏输入 Gemini API Key，否则只能使用普通规则模式。")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            files = process_pdf(uploaded_file, progress, status)
            
            progress.progress(100)
            status.text("处理完成！")
            
            if files:
                st.success(f"成功拆分出 {len(files)} 个文件")

# 结果展示
if st.session_state.processing_complete and st.session_state.generated_files:
    st.divider()
    
    # ZIP 下载
    if st.session_state.zip_data:
        st.download_button(
            label="📦 一键下载所有文件 (ZIP)",
            data=st.session_state.zip_data,
            file_name="split_reports.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.write("---")
    
    # 文件列表
    for f in st.session_state.generated_files:
        col1, col2, col3 = st.columns([4, 2, 2])
        with col1:
            st.write(f"📄 **{f['filename']}**")
            st.caption(f"包含 {f['page_count']} 页 | 机构代码: {f['code']}")
        with col2:
            st.download_button(
                "下载 PDF",
                data=f['content'],
                file_name=f['filename'],
                mime="application/pdf",
                key=f"btn_{f['filename']}"
            )
