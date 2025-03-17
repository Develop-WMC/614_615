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

# 嘗試從Streamlit secrets獲取API密鑰
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    GEMINI_API_KEY = ""  # 如果無法從secrets獲取，則使用空字串

# 初始化會話狀態
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = time.time()
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_file_for_preview' not in st.session_state:
    st.session_state.selected_file_for_preview = None

# 常見機構代碼
COMMON_CODES = {'OFS', 'WMG', 'WCL', 'DOL', 'LNI', 'DFW', 'DOR', 'ECY', 'WSP', 'DOH'}

# 添加日誌消息函數（僅內部使用，不顯示在UI上）
def add_log(message, level="info"):
    """添加日誌消息到會話狀態"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })

def check_rate_limit():
    """檢查API速率限制"""
    current_time = time.time()
    if current_time - st.session_state.last_reset_time >= 60:  # 60秒重置
        st.session_state.api_calls_count = 0
        st.session_state.last_reset_time = current_time
    
    if st.session_state.api_calls_count >= 60:  # 每分鐘最多60次
        remaining_time = 60 - (current_time - st.session_state.last_reset_time)
        raise Exception(f"超過API速率限制，請稍後再試。")
    
    st.session_state.api_calls_count += 1

# 文本提取代碼函數
def extract_code_from_text(text):
    """從文本中提取機構代碼"""
    if not text:
        return None
    
    # 標準化文本
    text = ' '.join(text.split()).upper()
    
    # 直接檢查高優先度代碼
    if 'OFS' in text:
        return 'OFS'
    
    if 'WMG' in text:
        return 'WMG'
    
    if 'WCL' in text:
        return 'WCL'
    
    # 常用模式
    patterns = [
        # Print Date後的三字母代碼
        r'PRINT DATE : \d{2} [A-Z]{3} \d{4}\s+([A-Z]{3})\s+PRINT TIME',
        # Agency或Code後的代碼
        r'(?:AGENCY|DEPT|CODE)[:\s]*([A-Z]{3})',
        # WA Code後的代碼
        r'WA CODE\s+([A-Z]{3})'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            code = matches[0]
            if code not in ['PDF', 'THE', 'AND', 'ALL', 'FOR']:
                return code
    
    return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def make_api_call(model, prompt, image):
    """調用Gemini API（帶重試）"""
    check_rate_limit()
    try:
        return model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(
            temperature=0
        ))
    except Exception as e:
        if "429" in str(e):
            time.sleep(5)
        raise e

def extract_code_with_ai(pdf_path, page_number, api_key, status_text):
    """使用AI識別代碼"""
    try:
        # 第一步：從文本中提取代碼
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        page_text = page.get_text()
        doc.close()
        
        # 使用文本規則提取代碼
        text_extracted_code = extract_code_from_text(page_text)
        
        # 如果規則方法找到高可信度代碼（如OFS、WMG），直接返回
        if text_extracted_code in ['OFS', 'WMG', 'WCL']:
            add_log(f"第 {page_number+1} 頁: 文本規則找到高可信度代碼 {text_extracted_code}")
            return {"code": text_extracted_code, "method": "text_rule", "confidence": "high", "text": page_text}
        
        # 如果沒有API密鑰，但有文本提取結果，則使用文本結果
        if not api_key and text_extracted_code:
            return {"code": text_extracted_code, "method": "text_rule", "confidence": "medium", "text": page_text}
        
        # 如果沒有API密鑰且沒有文本提取結果，返回未知
        if not api_key:
            return {"code": "UNK", "method": "text_rule", "confidence": "low", "text": page_text}
        
        # 使用AI模型識別
        page_image = convert_pdf_to_image(pdf_path, page_number)
        if not page_image:
            # 如果無法創建圖像，使用文本結果
            if text_extracted_code:
                return {"code": text_extracted_code, "method": "text_rule", "confidence": "medium", "text": page_text}
            return {"code": "UNK", "method": "text_rule", "confidence": "low", "text": page_text}
        
        # 配置Gemini API
        status_text.text(f"正在分析第 {page_number+1} 頁...")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Analyze this document image carefully.
        Task: Find the 3-letter agency/department code typically located at the top of the document.
        
        Important details:
        - Look specifically for 'OFS' or 'OFFICE OF FINANCIAL SERVICES'
        - Look specifically for 'WMG' after the PRINT DATE
        - Look specifically for 'WCL' in the document header
        - The code is exactly 3 capital letters
        - Common locations: header, top-right, or after "Agency:", "Code:", or "Dept:"
        - Common codes include: OFS (Office of Financial Services), WMG, WCL
        - Ignore common words like "THE", "AND", "ALL", "PDF", "REC"
        - If multiple codes appear, choose the most prominent one
        - If you see 'OFFICE OF FINANCIAL SERVICES' or 'OFS', return 'OFS'
        - If no valid code is found, return "UNK"
        
        RETURN EXACTLY (no explanation):
        {
            "code": "XXX"
        }
        where XXX is the 3-letter code you found or "UNK" if none found."""
        
        response = make_api_call(model, prompt, page_image)
        
        json_str = response.text
        
        # 解析JSON響應
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0]
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0]
        
        ai_results = json.loads(json_str.strip())
        ai_code = ai_results.get('code', 'UNK')
        
        # 第三步：結合兩種方法結果
        if ai_code == 'UNK' and text_extracted_code:
            return {"code": text_extracted_code, "method": "text_rule", "confidence": "medium", "text": page_text}
        
        if ai_code != 'UNK' and text_extracted_code and ai_code != text_extracted_code:
            # 如果兩種方法結果不一致
            if text_extracted_code in ['OFS', 'WMG', 'WCL']:
                return {"code": text_extracted_code, "method": "combined", "confidence": "high", "text": page_text}
            # 否則信任AI結果
            return {"code": ai_code, "method": "ai", "confidence": "high", "text": page_text}
        
        # 如果AI找到結果，信任它
        if ai_code != 'UNK':
            return {"code": ai_code, "method": "ai", "confidence": "high", "text": page_text}
        
        # 如果都沒找到
        return {"code": "UNK", "method": "combined", "confidence": "low", "text": page_text}
        
    except Exception as e:
        # 發生錯誤時，返回文本方法提取的代碼（如果有）
        if 'text_extracted_code' in locals() and text_extracted_code:
            return {"code": text_extracted_code, "method": "text_rule_fallback", "confidence": "low", "text": page_text if 'page_text' in locals() else ""}
        return {"code": "UNK", "method": "error", "confidence": "low", "text": page_text if 'page_text' in locals() else ""}

def convert_pdf_to_image(pdf_path, page_num):
    """將PDF頁面轉換為圖像 - 增強清晰度版本"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # 增加放大比例
        zoom = 6
        mat = fitz.Matrix(zoom, zoom)
        
        # 使用高DPI和無壓縮設置
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        
        # 使用PIL進一步控制圖像質量
        img = Image.open(io.BytesIO(img_data))
        
        return img
    except Exception as e:
        add_log(f"PDF轉圖像錯誤: {str(e)}", "error")
        return None
    finally:
        if doc:
            doc.close()

def is_summary_page(text):
    """檢查頁面是否為摘要頁"""
    if not text:
        return False
    summary_indicators = ["End of Report", "End of Reoprt", "Summary", "Grand Total"]
    return any(indicator in text for indicator in summary_indicators)

def determine_report_type(page_text):
    """根據內容確定報表類型"""
    if "Received Fees Report" in page_text:
        return {
            'report_num': '615',
            'format': 'MF'
        }
    elif "Outstanding" in page_text:
        return {
            'report_num': '614',
            'format': 'Outstanding'
        }
    return {
        'report_num': '615',
        'format': 'MF'
    }

def generate_filename(code, page_text):
    """生成適當的文件名"""
    report_info = determine_report_type(page_text)
    report_num = report_info['report_num']
    format_type = report_info['format']
    
    if report_num == '614':
        return f"Rpt 614-{code} {format_type}"
    else:
        return f"Rpt_{report_num}_{code}_MF"

def create_zip_buffer(generated_files):
    """創建包含所有文件的ZIP壓縮包"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in generated_files:
            zip_file.writestr(file['filename'], file['content'])
    zip_buffer.seek(0)
    return zip_buffer

def process_pdf(uploaded_file, progress_bar, status_text):
    """處理PDF文件"""
    temp_path = None
    
    try:
        # 重置處理狀態
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        
        # 檢查API密鑰
        if not GEMINI_API_KEY:
            st.warning("未配置Gemini API密鑰，將僅使用文本規則識別代碼。請聯繫管理員設置API密鑰以提高識別準確率。")
        
        # 清空生成的文件列表
        st.session_state.generated_files = []
        
        # 保存上傳的文件到臨時位置
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # 打開PDF並確定頁數
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        doc.close()
        
        # 用於存儲頁面的代碼信息
        page_codes = []
        
        # 第一步：識別所有頁面的代碼
        for page_num in range(total_pages):
            # 更新進度 - 這部分占總進度的一半
            progress = (page_num + 1) / (total_pages * 2)  # 總進度的50%
            progress_bar.progress(progress)
            status_text.text(f"識別第 {page_num + 1}/{total_pages} 頁的代碼...")
            
            # 從頁面提取文本
            doc = fitz.open(temp_path)
            page_text = doc[page_num].get_text()
            doc.close()
            
            # 檢查是否為摘要頁
            if is_summary_page(page_text) or page_num == total_pages - 1:
                page_codes.append({
                    'page_num': page_num,
                    'code': 'SUMMARY',
                    'text': page_text
                })
                continue
            
            # 提取代碼
            code_info = extract_code_with_ai(temp_path, page_num, GEMINI_API_KEY, status_text)
            
            code = code_info.get('code', 'UNK')
            if code in ['UNK', 'ALL']:
                code = 'UNKNOWN'
            
            # 存儲頁面代碼信息
            page_codes.append({
                'page_num': page_num,
                'code': code,
                'text': code_info.get('text', ''),
                'method': code_info.get('method', 'text_rule'),
                'confidence': code_info.get('confidence', 'medium')
            })
            
            # 暫停一下，確保界面更新
            time.sleep(0.1)
        
        # 第二步：識別連續相同代碼的頁面組
        page_groups = []
        current_group = []
        
        for i, page_info in enumerate(page_codes):
            if page_info['code'] == 'SUMMARY':
                # 摘要頁不計入分組
                if current_group:
                    page_groups.append(current_group)
                    current_group = []
                continue
                
            if not current_group:
                # 開始新的組
                current_group = [page_info]
            elif page_info['code'] == current_group[0]['code']:
                # 添加到當前組
                current_group.append(page_info)
            else:
                # 結束當前組並開始新的組
                page_groups.append(current_group)
                current_group = [page_info]
        
        # 添加最後一個組
        if current_group:
            page_groups.append(current_group)
        
        # 第三步：為每個組創建PDF文件，確保Adobe兼容性
        status_text.text("正在合併連續相同代碼的頁面...")
        
        for group_index, group in enumerate(page_groups):
            # 更新進度 - 這部分占總進度的另一半
            progress = 0.5 + ((group_index + 1) / (len(page_groups) * 2))  # 從50%到100%
            progress_bar.progress(progress)
            
            if not group:
                continue
            
            code = group[0]['code']
            first_page_text = group[0]['text']
            
            # 生成文件名
            base_filename = generate_filename(code, first_page_text)
            filename = f"{base_filename}.pdf"
            
            # 使用PyMuPDF (fitz) 創建新的PDF，而不是PyPDF2
            # 這樣可以確保更好的Adobe兼容性
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_output.close()
            
            # 打開源PDF
            source_doc = fitz.open(temp_path)
            # 創建新的PDF文檔
            output_doc = fitz.open()
            
            # 添加組中的所有頁面
            for page_info in group:
                page_num = page_info['page_num']
                output_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
            
            # 保存到臨時文件，使用更多Adobe兼容的設置
            output_doc.save(
                temp_output.name,
                garbage=4,  # 最大垃圾收集
                clean=True,  # 清理未使用的對象
                deflate=True,  # 使用deflate壓縮
                pretty=False,  # 不使用美化格式（減少文件大小）
                linear=True,  # 線性化PDF以便於Web瀏覽
            )
            
            output_doc.close()
            source_doc.close()
            
            # 讀取生成的文件內容
            with open(temp_output.name, 'rb') as file:
                file_content = file.read()
            
            # 添加到生成的文件列表
            st.session_state.generated_files.append({
                'filename': filename,
                'content': file_content,
                'pages': [p['page_num'] for p in group],
                'code': code,
                'page_count': len(group),
                'method': group[0].get('method', 'text_rule'),
                'confidence': group[0].get('confidence', 'medium')
            })
            
            # 刪除臨時文件
            try:
                os.remove(temp_output.name)
            except:
                pass
            
            # 暫停一下，確保界面更新
            time.sleep(0.1)
        
        # 處理完成
        st.session_state.processing_complete = True
        
        # 創建ZIP數據 - 將ZIP數據保存到會話狀態
        if st.session_state.generated_files:
            st.session_state.zip_data = create_zip_buffer(st.session_state.generated_files)
        
        return st.session_state.generated_files
    
    except Exception as e:
        st.error(f"處理PDF時出錯: {str(e)}")
        return []
    
    finally:
        # 刪除臨時文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def set_preview_file(file_index):
    """設置要預覽的文件"""
    st.session_state.selected_file_for_preview = file_index

# Streamlit UI
st.title("PDF報表拆分工具")
st.write("此工具可以將多頁PDF報表按機構代碼拆分成單獨的文件")

# 側邊欄：使用說明
with st.sidebar:
    st.header("使用說明")
    st.markdown("1. 上傳PDF文件")
    st.markdown("2. 點擊「處理PDF」按鈕")
    st.markdown("3. 等待處理完成")
    st.markdown("4. 下載拆分後的文件")
    
    # 顯示功能說明
    st.markdown("---")
    st.subheader("功能說明")
    st.markdown("- 連續的相同代碼頁面會自動合併為一個PDF文件")
    st.markdown("- 不同代碼的頁面會分開成獨立的PDF文件")
    st.markdown("- 摘要頁面會被自動忽略")
    
    # 顯示瀏覽器兼容性提示
    st.markdown("---")
    st.subheader("兼容性說明")
    st.info("如果您在Adobe查看器中遇到顯示問題，請嘗試使用Chrome或Edge瀏覽器打開生成的文件")
    
    # 顯示API配置狀態
    st.markdown("---")
    st.subheader("系統狀態")
    if GEMINI_API_KEY:
        st.success("✅ Gemini API已配置")
    else:
        st.warning("⚠️ Gemini API未配置")
        st.info("請聯繫管理員配置API以提高識別準確率")

# 上傳文件
uploaded_file = st.file_uploader("選擇PDF文件", type="pdf")

if uploaded_file is not None:
    # 顯示文件信息
    st.info(f"已上傳: {uploaded_file.name} ({round(uploaded_file.size/1024, 1)} KB)")
    
    # 顯示預覽
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        doc = fitz.open(temp_path)
        preview_page = 0
        page = doc[preview_page]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
        img_data = pix.tobytes("png")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("文件首頁預覽:")
        with col2:
            st.image(img_data)
        
        doc.close()
        os.remove(temp_path)
    except Exception as e:
        st.warning(f"無法顯示預覽")

    # 處理按鈕
    process_button = st.button("處理PDF", key="process_button", use_container_width=True)
    
    if process_button:
        # 重置處理狀態
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        st.session_state.selected_file_for_preview = None
        
        # 創建進度顯示區域
        progress_container = st.container()
        with progress_container:
            st.write("處理進度:")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("準備處理...")
            
            # 處理PDF
            generated_files = process_pdf(uploaded_file, progress_bar, status_text)
            
            # 完成處理
            progress_bar.progress(1.0)
            status_text.text(f"處理完成! 已生成 {len(generated_files)} 個文件。")
            
            if len(generated_files) > 0:
                st.success("✓ 處理成功! 請點擊下方的下載按鈕獲取處理後的文件。")
                st.info("💡 提示: 如果在Adobe中查看文件有問題，請嘗試使用Chrome或Edge打開。")

# 如果處理已完成且有文件，顯示結果
if st.session_state.processing_complete and st.session_state.generated_files:
    st.markdown("---")
    st.subheader(f"處理結果 (共 {len(st.session_state.generated_files)} 個文件)")
    
    # 顯示ZIP下載按鈕
    if st.session_state.zip_data:
        st.download_button(
            label=f"下載所有文件 (ZIP包含 {len(st.session_state.generated_files)} 個文件)",
            data=st.session_state.zip_data,
            file_name="processed_files.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # 按機構代碼分組
    grouped_files = {}
    for i, file in enumerate(st.session_state.generated_files):
        code = file.get('code', 'UNK')
        if code not in grouped_files:
            grouped_files[code] = []
        grouped_files[code].append((i, file))
    
    # 分組顯示
    tabs = st.tabs([f"{code} ({len(files)})" for code, files in grouped_files.items()])
    
    for i, (code, files) in enumerate(grouped_files.items()):
        with tabs[i]:
            # 文件表格
            st.write("點擊檔案名稱來預覽文件:")
            
            # 文件列表
            for file_idx, file in files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # 使用按鈕作為點擊預覽的方式
                    page_range = f"第{min(file['pages'])+1}-{max(file['pages'])+1}頁" if len(file['pages']) > 1 else f"第{file['pages'][0]+1}頁"
                    if st.button(f"{file['filename']} ({page_range}, 共{file['page_count']}頁)", key=f"file_{file_idx}", use_container_width=True):
                        set_preview_file(file_idx)
                with col2:
                    st.download_button(
                        label="下載",
                        data=file['content'],
                        file_name=file['filename'],
                        mime="application/pdf",
                        key=f"download_{file_idx}"
                    )
            
            # 顯示統計信息
            st.info(f"機構代碼 {code}: {len(files)} 個文件")
    
    # 如果選擇了要預覽的文件，顯示它
    if st.session_state.selected_file_for_preview is not None:
        file_idx = st.session_state.selected_file_for_preview
        if 0 <= file_idx < len(st.session_state.generated_files):
            file = st.session_state.generated_files[file_idx]
            
            st.markdown("---")
            st.subheader(f"預覽: {file['filename']}")
            
            # 使用PIL顯示PDF預覽
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file['content'])
                temp_path = tmp_file.name
            
            try:
                doc = fitz.open(temp_path)
                # 如果文件頁數超過5頁，只預覽前5頁
                max_preview_pages = min(5, doc.page_count)
                
                st.write(f"預覽前 {max_preview_pages} 頁 (共 {doc.page_count} 頁):")
                
                for page_idx in range(max_preview_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
                    img_data = pix.tobytes("png")
                    st.image(img_data, caption=f"第 {page_idx+1} 頁")
                
                doc.close()
            except Exception as e:
                st.error(f"顯示預覽時出錯")
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass

# 頁腳
st.markdown("---")
st.markdown("© 2025 PDF報表拆分工具")
