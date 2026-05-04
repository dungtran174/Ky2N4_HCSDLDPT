import streamlit as st
import time
import os

import numpy as np
from scipy.io import wavfile
import tempfile

# Import class AudioSearchEngine từ Giai đoạn 4
from search_engine import AudioSearchEngine

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Multimedia Audio Search", page_icon="🎵", layout="wide")

# ==========================================
# 1. KHỞI TẠO HỆ THỐNG (CACHE)
# ==========================================
@st.cache_resource
def load_engine():
    DB_PARAMS = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "123456",
        "host": "localhost",
        "port": "5432"
    }
    return AudioSearchEngine(DB_PARAMS)

engine = load_engine()



# ==========================================
# 3. SIDEBAR & GIAO DIỆN CHÍNH
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/11836/11836093.png", width=100)
st.sidebar.title("Thông tin Đồ án")
st.sidebar.markdown("""
**Nhóm:** 05  
**Môn học:** Hệ CSDL Đa phương tiện  


---
### 📊 Quy mô CSDL
* **Tổng số files:** 510 audio  
* **Đặc trưng Giọng nói:** Vector 32 chiều (Euclidean)  
* **Đặc trưng Nội dung:** Vector 384 chiều (Cosine)
""")

st.title("🎵 Audio Search Engine")
st.markdown("Hệ thống truy vấn thông tin âm thanh dựa trên đặc tính vật lý (Voice) và ngữ nghĩa (Content).")


# ==========================================
# 5. KHU VỰC UPLOAD & TRUY VẤN
# ==========================================
st.subheader("1. Tải lên Audio Truy vấn (Query Audio)")
uploaded_file = st.file_uploader("Chọn file âm thanh (.wav)", type=['wav'])

if uploaded_file is not None:
    # Lưu file tạm để xử lý
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_audio_path = tmp_file.name

    st.audio(tmp_audio_path, format="audio/wav")
    


    # Nút thực hiện tìm kiếm
    if st.button("🚀 Bắt đầu Tìm kiếm (Search)", use_container_width=True):
        start_time = time.time()
        
        with st.spinner("Đang trích xuất đặc trưng và truy vấn CSDL..."):
            # 1. Trích xuất
            voice_vec, content_vec, transcript = engine.process_query(tmp_audio_path)
            
            # 2. Truy vấn DB
            top_voices = engine.search_voice(voice_vec, top_k=3)
            top_contents = engine.search_content(content_vec, top_k=3)
            
        query_time = time.time() - start_time
        st.success(f"Truy vấn hoàn tất trong {query_time:.2f} giây! Văn bản nhận diện được: **'{transcript}'**")
        
        st.divider()
        
        # ==========================================
        # 6. HIỂN THỊ KẾT QUẢ TÌM KIẾM
        # ==========================================
        st.subheader("2. Kết quả Tìm kiếm (Top 3)")
        
        # PHẦN 1: VOICE
        st.markdown("<h4 style='color: #ff4b4b;'>🗣️ Top 3 Giọng nói giống nhất</h4>", unsafe_allow_html=True)
        for rank, (filename, score, text) in enumerate(top_voices, 1):
            with st.container():
                st.markdown(f"**#{rank}: {filename}**")
                # Score của L2 thường nhỏ, ta normalize hiển thị %
                display_score = min(score * 100, 100.0) 
                st.markdown(f"**Voice Similarity:** {display_score:.2f}%")
                
                # File phát nhạc
                res_path = os.path.join("data", filename)
                if os.path.exists(res_path):
                    st.audio(res_path)
                else:
                    st.warning("File không tồn tại trong thư mục data/")
                    
        st.divider()
                    
        # PHẦN 2: CONTENT
        st.markdown("<h4 style='color: #1f77b4;'>📝 Top 3 Nội dung giống nhất</h4>", unsafe_allow_html=True)
        for rank, (filename, score, text) in enumerate(top_contents, 1):
            with st.container():
                st.markdown(f"**#{rank}: {filename}**")
                display_score = min(score * 100, 100.0)
                st.markdown(f"**Content Similarity:** {display_score:.2f}%")
                
                st.markdown(f"*💬 Trích đoạn:* \"{text[:150]}...\"")
                
                with st.expander("Xem toàn bộ Transcript"):
                    st.write(text)
                
                # Bỏ hiển thị Audio cho phần nội dung
                        
    # Xóa file tạm
    os.unlink(tmp_audio_path)
