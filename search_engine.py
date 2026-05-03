import psycopg2
import json
import whisper
from sentence_transformers import SentenceTransformer
import numpy as np

# Tái sử dụng logic trích xuất từ Giai đoạn 1
from feature_extractor import extract_features

class AudioSearchEngine:
    def __init__(self, db_params):
        """
        Khởi tạo Search Engine, load sẵn các model AI để tránh việc load lại nhiều lần
        """
        self.db_params = db_params
        
        print("Đang tải model Whisper (Base) cho việc nhận diện giọng nói...")
        # Sử dụng model 'base' để cân bằng giữa tốc độ và độ chính xác
        self.whisper_model = whisper.load_model("base")
        
        print("Đang tải model Sentence-Transformers cho việc phân tích ngữ nghĩa...")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load scaler params để chuẩn hóa Query Vector
        try:
            with open('scaler_params.json', 'r') as f:
                self.scaler_params = json.load(f)
            print("Đã tải Scaler Params (Z-score) thành công!")
        except Exception as e:
            print("Cảnh báo: Không tìm thấy scaler_params.json, sẽ không chuẩn hóa.")
            self.scaler_params = None
        
    def process_query(self, audio_path):
        """
        Quy trình xử lý Audio đầu vào (Query)
        Trích xuất cả đặc trưng giọng nói (Voice) và ngữ nghĩa (Content)
        """
        print(f"\n--- ĐANG XỬ LÝ TRUY VẤN: {audio_path} ---")
        
        # ==========================================
        # NHÁNH 1: VOICE (Acoustic Features)
        # ==========================================
        print("1. Trích xuất đặc trưng âm thanh (Voice)...")
        # Gọi hàm extract_features từ Giai đoạn 1 (numpy/scipy)
        voice_dict = extract_features(audio_path)
        
        # Gom các giá trị dict thành 1 vector list 32 chiều theo đúng thứ tự
        voice_vector = []
        voice_cols = []
        for i in range(1, 14):
            voice_vector.extend([voice_dict[f'mfcc_{i}_mean'], voice_dict[f'mfcc_{i}_std']])
            voice_cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
        
        voice_vector.extend([
            voice_dict['zcr_mean'], voice_dict['zcr_std'],
            voice_dict['energy_mean'], voice_dict['energy_std'],
            voice_dict['centroid_mean'], voice_dict['centroid_std']
        ])
        voice_cols.extend(['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'centroid_mean', 'centroid_std'])
        
        # Chuẩn hóa Z-Score ngay tại đây nếu có params
        if self.scaler_params:
            for i, col in enumerate(voice_cols):
                mean_val = self.scaler_params['mean'][col]
                std_val = self.scaler_params['std'][col]
                if std_val != 0:
                    voice_vector[i] = (voice_vector[i] - mean_val) / std_val
        
        # ==========================================
        # NHÁNH 2: CONTENT (Semantic Features)
        # ==========================================
        print("2. Chuyển đổi giọng nói thành văn bản (Speech-to-Text)...")
        transcribe_result = self.whisper_model.transcribe(audio_path)
        transcript = transcribe_result["text"].strip().lower()
        print(f"   => Transcript: '{transcript}'")
        
        print("3. Trích xuất đặc trưng ngữ nghĩa (Text Embeddings)...")
        content_vector = self.st_model.encode(transcript).tolist()
        
        return voice_vector, content_vector, transcript

    def search_voice(self, voice_vector, top_k=3):
        """
        Truy vấn Top 3 Giọng nói giống nhất sử dụng Euclidean Distance (Toán tử <->)
        """
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        
        # Dùng toán tử <-> để tính Khoảng cách Euclidean.
        # Khoảng cách càng nhỏ (gần 0) thì càng giống nhau.
        # Để quy về điểm Tương đồng (Score) giảm dần, ta dùng công thức: Score = 1 / (1 + Distance)
        query = """
            SELECT m.filename, 
                   1 / (1 + (v.voice_vector <-> %s::vector) / 10.0) AS similarity_score,
                   m.transcript
            FROM audio_vectors v
            JOIN audio_metadata m ON v.id = m.id
            ORDER BY similarity_score DESC
            LIMIT %s;
        """
        # Chuyển list Python thành chuỗi dạng '[0.1, 0.2, ...]' để PostgreSQL hiểu
        cur.execute(query, (json.dumps(voice_vector), top_k))
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        return results

    def search_content(self, content_vector, top_k=3):
        """
        Truy vấn Top 3 Nội dung giống nhất sử dụng Cosine Similarity (Toán tử <=>)
        """
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        
        # Trong pgvector, toán tử <=> trả về Cosine Distance (Khoảng cách Cosine)
        # Cosine Distance = 1 - Cosine Similarity
        # Do đó, để lấy Độ tương đồng (Similarity Score), ta lấy 1 - Cosine Distance
        query = """
            SELECT m.filename, 
                   1 - (v.content_vector <=> %s::vector) AS similarity_score,
                   m.transcript
            FROM audio_vectors v
            JOIN audio_metadata m ON v.id = m.id
            ORDER BY similarity_score DESC
            LIMIT %s;
        """
        cur.execute(query, (json.dumps(content_vector), top_k))
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        return results

# ==========================================
# KHỐI TEST CODE
# ==========================================
if __name__ == "__main__":
    DB_PARAMS = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "123456",
        "host": "localhost",
        "port": "5432"
    }
    
    # Lưu ý: Cần cài thêm thư viện openai-whisper nếu chưa có: pip install openai-whisper
    engine = AudioSearchEngine(DB_PARAMS)
    
    # Ví dụ dùng 1 file trong tập test
    # test_file = "test/1file_nao_do.wav" 
    # voice_vec, content_vec, text = engine.process_query(test_file)
    
    # top_voices = engine.search_voice(voice_vec)
    # top_contents = engine.search_content(content_vec)
    
    # In kết quả ...
