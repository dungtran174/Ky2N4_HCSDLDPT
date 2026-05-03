import pandas as pd
import psycopg2
import json

def prepare_data():
    """
    Đọc và Merge dữ liệu từ 3 file CSV
    """
    print("Đang đọc dữ liệu từ các file CSV...")
    
    # 1. Đọc metadata
    df_meta = pd.read_csv('metadata.csv')
    
    # 2. Đọc đặc trưng giọng nói (Giai đoạn 1) đã được chuẩn hóa (Z-Score)
    df_voice = pd.read_csv('audio_voice_features_scaled.csv')
    # Liệt kê đúng 32 cột đặc trưng
    voice_cols = []
    for i in range(1, 14):
        voice_cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    voice_cols.extend(['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'centroid_mean', 'centroid_std'])
    
    # Gộp 32 cột thành một list duy nhất để nạp vào pgvector
    df_voice['voice_vector'] = df_voice[voice_cols].values.tolist()
    df_voice = df_voice[['file_id', 'voice_vector']]
    
    # 3. Đọc đặc trưng nội dung (Giai đoạn 2)
    df_content = pd.read_csv('audio_content_features.csv')
    # Cột content_vector đang là chuỗi JSON, parse ngược lại thành list Python
    df_content['content_vector'] = df_content['content_vector'].apply(json.loads)
    df_content = df_content[['file_id', 'content_vector']]
    
    # 4. Merge dữ liệu
    print("Đang nối (Merge) dữ liệu dựa trên file_id...")
    df_final = df_meta.merge(df_voice, on='file_id', how='inner')
    df_final = df_final.merge(df_content, on='file_id', how='inner')
    
    print(f"Đã merge thành công! Tổng số bản ghi sẵn sàng nạp: {len(df_final)}")
    return df_final

def setup_database_and_load(df):
    """
    Kết nối DB, tạo bảng, tạo index và nạp dữ liệu
    """
    # Thông tin kết nối PostgreSQL (dựa trên thông tin bạn cung cấp)
    db_params = {
        "dbname": "postgres", # Mặc định thường là postgres
        "user": "postgres",   # User mặc định
        "password": "123456", # Mật khẩu bạn cung cấp
        "host": "localhost",
        "port": "5432"
    }
    
    try:
        # Kết nối tới database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        print("Kết nối PostgreSQL thành công!")
        
        # 1. Kích hoạt extension pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Xoá bảng cũ (nếu có) để chạy lại script dễ dàng
        cur.execute("DROP TABLE IF EXISTS audio_vectors CASCADE;")
        cur.execute("DROP TABLE IF EXISTS audio_metadata CASCADE;")
        
        # 3. Tạo bảng audio_metadata
        print("Đang tạo các bảng CSDL...")
        cur.execute("""
            CREATE TABLE audio_metadata (
                id SERIAL PRIMARY KEY,
                file_id VARCHAR UNIQUE NOT NULL,
                filename VARCHAR,
                filepath VARCHAR,
                transcript TEXT
            );
        """)
        
        # 4. Tạo bảng audio_vectors (Quan hệ 1-1 qua cột id)
        cur.execute("""
            CREATE TABLE audio_vectors (
                id INTEGER PRIMARY KEY REFERENCES audio_metadata(id) ON DELETE CASCADE,
                voice_vector vector(32),
                content_vector vector(384)
            );
        """)
        
        # 5. Tạo chỉ mục HNSW tối ưu hóa tìm kiếm
        print("Đang xây dựng Index HNSW cho các Vector...")
        # Sử dụng Euclidean (L2) cho voice_vector
        cur.execute("CREATE INDEX ON audio_vectors USING hnsw (voice_vector vector_l2_ops);")
        # Sử dụng Cosine Similarity cho content_vector
        cur.execute("CREATE INDEX ON audio_vectors USING hnsw (content_vector vector_cosine_ops);")
        
        # 6. Nạp dữ liệu
        print("Bắt đầu nạp dữ liệu vào bảng...")
        for idx, row in df.iterrows():
            # Chèn metadata
            cur.execute("""
                INSERT INTO audio_metadata (file_id, filename, filepath, transcript)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (row['file_id'], row['filename'], row['filepath'], row['transcript']))
            
            # Lấy ID tự sinh của bảng metadata
            inserted_id = cur.fetchone()[0]
            
            # Chèn vector vào bảng audio_vectors với ID vừa lấy
            # Note: psycopg2 tự động chuyển đổi list Python thành format pgvector cần
            cur.execute("""
                INSERT INTO audio_vectors (id, voice_vector, content_vector)
                VALUES (%s, %s, %s);
            """, (inserted_id, row['voice_vector'], row['content_vector']))
            
            if (idx + 1) % 50 == 0:
                print(f"Đã nạp {idx + 1}/{len(df)} bản ghi...")
                
        # Commit giao dịch
        conn.commit()
        print("Nạp dữ liệu hoàn tất thành công!")
        
    except Exception as e:
        print(f"Lỗi thao tác Database: {e}")
        if 'conn' in locals():
            conn.rollback() # Hoàn tác nếu có lỗi
    finally:
        # Đóng kết nối
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
            print("Đã đóng kết nối CSDL.")

if __name__ == "__main__":
    df_data = prepare_data()
    setup_database_and_load(df_data)
