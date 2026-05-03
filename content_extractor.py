import pandas as pd
import json
import re
from sentence_transformers import SentenceTransformer

def preprocess_text(text):
    """
    Tiền xử lý văn bản:
    - Chuyển về chữ thường (lowercase)
    - Loại bỏ các ký tự đặc biệt không mang nghĩa (giữ lại chữ, số và khoảng trắng cơ bản)
    - Loại bỏ khoảng trắng thừa
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Chuyển chữ thường
    text = text.lower()
    
    # Loại bỏ ký tự đặc biệt (chỉ giữ lại a-z, 0-9 và khoảng trắng)
    # Lưu ý: Tuỳ vào bài toán, có thể giữ lại dấu chấm câu nếu model cần ngữ cảnh câu, 
    # nhưng all-MiniLM-L6-v2 khá linh hoạt, ta có thể làm sạch cơ bản.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_content_features(input_csv, output_csv):
    print("Đang tải model all-MiniLM-L6-v2 (có thể mất chút thời gian nếu chạy lần đầu)...")
    # Khởi tạo model Sentence Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Đọc dữ liệu từ file metadata
    df = pd.read_csv(input_csv)
    
    all_features = []
    
    print(f"Bắt đầu trích xuất đặc trưng văn bản cho {len(df)} files...")
    for idx, row in df.iterrows():
        file_id = row['file_id']
        filename = row['filename']
        transcript = row.get('transcript', "")
        
        # Tiền xử lý
        cleaned_text = preprocess_text(transcript)
        
        # Xử lý ngoại lệ nếu transcript trống
        if not cleaned_text:
            print(f"Cảnh báo: Transcript trống hoặc lỗi ở file {filename}. Gán vector zero.")
            # Tạo vector 0 với 384 chiều
            vector = [0.0] * 384
        else:
            # Mã hoá văn bản thành vector 384 chiều
            # Trả về numpy array
            vector_np = model.encode(cleaned_text)
            # Chuyển numpy array thành list số thực Python
            vector = vector_np.tolist()
            
        # Chuyển list thành chuỗi JSON để dễ lưu vào DB (ví dụ: PostgreSQL dạng JSONB/Vector)
        vector_json = json.dumps(vector)
        
        all_features.append({
            'file_id': file_id,
            'filename': filename,
            'content_vector': vector_json
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Đã xử lý {idx + 1}/{len(df)} transcripts.")
            
    # Tạo DataFrame và lưu kết quả
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(output_csv, index=False)
    print(f"Hoàn thành! Đã lưu {len(df_features)} vectors vào {output_csv}")

if __name__ == '__main__':
    # File metadata hiện tại của hệ thống
    input_metadata = 'metadata.csv' 
    output_features = 'audio_content_features.csv'
    
    process_content_features(input_metadata, output_features)
