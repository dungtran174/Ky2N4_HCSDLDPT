# 🎵 Hybrid Audio Search Engine (CBIR)
**Môn học:** Hệ Cơ sở Dữ liệu Đa phương tiện | **Nhóm:** 05

---

## 1. Tổng quan Dự án

Hệ thống truy vấn âm thanh lai (Hybrid Audio Retrieval) cho phép người dùng upload một file `.wav` và tìm kiếm ra Top-3 file âm thanh giống nhất trong cơ sở dữ liệu theo **hai tiêu chí song song**:

| Chiều tìm kiếm | Phương pháp | Đặc trưng | Độ đo |
|---|---|---|---|
| **Giọng nói** (Voice) | Xử lý tín hiệu số | Vector 32 chiều (MFCC, ZCR, Energy, Centroid) | Euclidean Distance |
| **Nội dung** (Content) | Deep Learning | Vector 384 chiều (Text Embedding) | Cosine Similarity |

---

## 2. Bộ Dữ liệu

- **Nguồn:** Các bài diễn thuyết TED Talks định dạng `.wav`
- **Quy mô:** 510 file audio (tập train) + 17 file (tập test)
- **Định dạng:** PCM WAV, Mono, 16kHz
- **Nội dung:** Tiếng Anh, dài từ 1 - 3 phút/file, nhiều người nói khác nhau

### Cấu trúc thư mục
```
btl/
├── data/                          # 510 file audio train
├── test/                          # 17 file audio test
├── audio_voice_features.csv       # Đặc trưng giọng nói (raw)
├── audio_voice_features_scaled.csv# Đặc trưng giọng nói (Z-Score)
├── audio_content_features.csv     # Đặc trưng nội dung (embeddings)
├── metadata.csv                   # Thông tin file + transcript
├── scaler_params.json             # Tham số chuẩn hóa Z-Score (QUAN TRỌNG)
├── feature_extractor.py           # Giai đoạn 1: Trích xuất Voice
├── content_extractor.py           # Giai đoạn 2: Trích xuất Content
├── normalize_features.py          # Chuẩn hóa StandardScaler
├── database_loader.py             # Nạp dữ liệu vào PostgreSQL
├── search_engine.py               # Class tìm kiếm chính
└── app.py                         # Giao diện Streamlit
```

---

## 3. Đặc trưng Giọng nói (Voice Features) — Vector 32 chiều

### 3.1. Quy trình Tiền xử lý Tín hiệu

```
File WAV → Pre-emphasis → Framing → Windowing (Hamming) → FFT → Mel Filterbank → DCT → MFCC
```

**Bước 1: Pre-emphasis** — Tăng cường thành phần tần số cao bị suy giảm tự nhiên:
$$y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97$$

**Bước 2: Framing** — Cắt tín hiệu thành các khung nhỏ chồng lấp:
- Frame size: 25ms (~400 samples tại 16kHz)
- Frame shift: 10ms (~160 samples)

**Bước 3: Hamming Window** — Giảm nhiễu biên khung:
$$w[n] = 0.54 - 0.46 \cdot \cos\!\left(\frac{2\pi n}{N-1}\right)$$

**Bước 4: FFT → Mel Filterbank → Log → DCT = MFCC**

### 3.2. Bảng 32 Đặc trưng Giọng nói

| STT | Tên đặc trưng | Ý nghĩa | Số chiều |
|---|---|---|---|
| 1-26 | `mfcc_1_mean` → `mfcc_13_std` | Biểu diễn âm sắc (Timbre), cấu trúc phổ giọng nói | 26 |
| 27 | `zcr_mean` | Tỷ lệ tín hiệu cắt qua 0 — phân biệt âm hữu thanh/vô thanh | 1 |
| 28 | `zcr_std` | Mức độ biến động của ZCR | 1 |
| 29 | `energy_mean` | Năng lượng trung bình — đo độ to/nhỏ giọng | 1 |
| 30 | `energy_std` | Mức độ thay đổi năng lượng | 1 |
| 31 | `centroid_mean` | Tần số trọng tâm phổ — giọng trầm/bổng (Hz) | 1 |
| 32 | `centroid_std` | Mức độ biến động của tần số | 1 |

### 3.3. Công thức tính từng đặc trưng

**MFCC (Mel-Frequency Cepstral Coefficients):**
$$\text{Mel}(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$
$$\text{MFCC}_k = \sum_{m=1}^{M} \log S_m \cdot \cos\!\left[k\left(m - \frac{1}{2}\right)\frac{\pi}{M}\right]$$

**Zero Crossing Rate (ZCR):**
$$\text{ZCR} = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{1}[x[n] \cdot x[n-1] < 0]$$

**RMS Energy:**
$$E = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

**Spectral Centroid:**
$$C = \frac{\sum_k f_k \cdot |X[k]|}{\sum_k |X[k]|}$$

### 3.4. Tại sao chọn 32 đặc trưng này?

- **MFCC:** Bắt chước cơ chế nghe của tai người (thang Mel phi tuyến). Là đặc trưng tiêu chuẩn trong mọi hệ thống nhận dạng giọng nói (ASR, Speaker ID).
- **ZCR:** Phân biệt âm vô thanh (s, f, sh) với âm hữu thanh (a, e, i). Phản ánh đặc tính ngữ âm của người nói.
- **Energy:** Người nói to/nhỏ khác nhau tạo ra profile năng lượng đặc trưng.
- **Centroid:** Giọng nữ thường có centroid cao hơn giọng nam vì tần số cơ bản (F0) cao hơn.

### 3.5. Chuẩn hóa Z-Score (StandardScaler)

**Vấn đề:** Centroid có giá trị hàng ngàn Hz trong khi MFCC chỉ vài chục → Euclidean bị Centroid "áp đảo".

**Giải pháp:** Chuẩn hóa mỗi cột về Mean=0, Std=1:
$$z = \frac{x - \mu}{\sigma}$$

Các tham số $\mu$ và $\sigma$ được tính trên toàn bộ 510 file và lưu vào `scaler_params.json`. **Khi truy vấn, file audio mới bắt buộc phải được chuẩn hóa bằng đúng cùng bộ tham số này.**

---

## 4. Đặc trưng Nội dung (Content Features) — Vector 384 chiều

### 4.1. Quy trình

```
File WAV → Whisper (Speech-to-Text) → Transcript → all-MiniLM-L6-v2 → Vector 384D
```

**Bước 1: Speech-to-Text bằng OpenAI Whisper**
- Model: `whisper-base` (74M parameters)
- Cơ chế: Encoder-Decoder Transformer, nhận đầu vào là log-Mel spectrogram 80 chiều
- Output: Chuỗi văn bản tiếng Anh đầy đủ của bài nói

**Bước 2: Text Embedding bằng Sentence-Transformers**
- Model: `all-MiniLM-L6-v2` (22M parameters, 384 chiều)
- Cơ chế: BERT-based encoder, ánh xạ toàn bộ đoạn văn bản sang một điểm trong không gian ngữ nghĩa 384 chiều
- Các câu có **cùng chủ đề** sẽ nằm **gần nhau** trong không gian này

### 4.2. Tại sao dùng Embedding thay vì TF-IDF/Keyword?

| Tiêu chí | TF-IDF / Keyword | Sentence Embedding |
|---|---|---|
| Phương pháp | Đếm tần suất từ | Học ngữ nghĩa từ ngữ cảnh |
| "tiền" vs "thu nhập" | ❌ Không khớp (0%) | ✅ Nhận ra đồng nghĩa (~90%) |
| Lỗi chính tả STT | ❌ Nhạy cảm cao | ✅ Chịu lỗi tốt |
| Độ chính xác ngữ nghĩa | Thấp | Cao (State-of-the-Art) |

### 4.3. Tại sao dùng Cosine Similarity cho Content?

Vector Embedding được thiết kế để **phương hướng (direction)** mang ý nghĩa, không phải độ lớn (magnitude). Hai văn bản cùng chủ đề sẽ có vector chỉ về cùng một hướng trong không gian 384 chiều:

$$\text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}||\vec{B}|} = \cos\theta$$

- $\theta = 0°$: Hai văn bản cùng chủ đề hoàn toàn → Score = 100%
- $\theta = 90°$: Hai văn bản không liên quan → Score = 0%

Trong pgvector, toán tử `<=>` trả về **Cosine Distance** (= 1 - Cosine Similarity). Do đó điểm tương đồng hiển thị được tính:
$$\text{Score}_\text{content} = 1 - \text{CosineDistance}(\vec{q}, \vec{d})$$

---

## 4.5. Độ đo Tương đồng Giọng nói (Voice Similarity Score)

### Công thức gốc: Khoảng cách Euclidean (L2 Distance)

Sau khi chuẩn hóa Z-Score, hai vector giọng nói $\vec{q}$ (query) và $\vec{d}$ (database) được so sánh bằng:

$$d(\vec{q}, \vec{d}) = \sqrt{\sum_{i=1}^{32} (q_i - d_i)^2}$$

Trong pgvector, toán tử `<->` tính chính xác khoảng cách L2 này.

### Vấn đề: Khoảng cách không có giới hạn trên

Khoảng cách Euclidean có giá trị từ $[0, +\infty)$:
- Khoảng cách = 0 → giống hệt nhau (chính file đó)
- Khoảng cách = 8 → khác biệt đáng kể

Người dùng quen nhìn kết quả theo thang **0% - 100%**, không phải con số khoảng cách thô. Do đó cần biến đổi.

### Giải pháp: Inverse Distance Transformation + Scaling

$$\text{Score}_\text{voice} = \frac{1}{1 + \dfrac{d(\vec{q}, \vec{d})}{s}}$$

Trong đó:
- $d(\vec{q}, \vec{d})$: Khoảng cách Euclidean giữa 2 vector (sau Z-Score)
- $s = 10.0$: Hệ số tỷ lệ (scaling factor / hyperparameter)

**Phân tích công thức:**

| Khoảng cách $d$ | Score (s=10) | Ý nghĩa |
|---|---|---|
| 0 | 100% | Chính xác file đó |
| 2 | 83.3% | Rất giống |
| 5 | 66.7% | Khá giống |
| 10 | 50% | Tương đồng trung bình |
| 20 | 33.3% | Ít giống |
| 50 | 16.7% | Gần như khác biệt |

**Tại sao có hằng số `+ 1` ở mẫu số?**  
Để tránh lỗi chia cho 0 khi $d = 0$ (file giống hệt nhau). Phép cộng 1 cũng đảm bảo giá trị tối đa của Score luôn bằng đúng 1.0 (100%).

**Tại sao chọn $s = 10.0$?**  
Sau khi chuẩn hóa Z-Score, khoảng cách Euclidean giữa các file trong CSDL thường dao động từ 3 đến 15. Hệ số $s = 10$ giúp "trải phẳng" dải điểm ra vùng 40%-80%, tránh tình trạng điểm bị dồn cục ở 1-2% (khi chưa chuẩn hóa và $s = 1$).

**Tính chất quan trọng của công thức:**  
Hàm $f(d) = \frac{1}{1 + d/s}$ là hàm **đơn điệu giảm** (monotonically decreasing):
- Khoảng cách nhỏ hơn → Score cao hơn
- **Thứ tự xếp hạng Top-1, Top-2, Top-3 KHÔNG thay đổi** so với sắp xếp theo khoảng cách thuần túy
- Thay đổi $s$ chỉ ảnh hưởng đến giá trị số hiển thị, không ảnh hưởng kết quả tìm kiếm



## 5. Cơ sở dữ liệu PostgreSQL + pgvector

### 5.1. Lý do chọn Mô hình Quan hệ

PostgreSQL được chọn vì:
1. **Hỗ trợ extension `pgvector`**: Lưu trữ và tìm kiếm vector tương đồng ngay trong SQL
2. **Tính toàn vẹn dữ liệu**: Ràng buộc khóa ngoại đảm bảo tính nhất quán
3. **ACID**: An toàn khi nạp 510 bản ghi song song
4. **Index HNSW**: Tìm kiếm vector tốc độ cao (O(log n) thay vì O(n))

### 5.2. Thiết kế Schema

```sql
-- Cài extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng 1: Thông tin mô tả
CREATE TABLE audio_metadata (
    id       SERIAL PRIMARY KEY,
    file_id  VARCHAR UNIQUE NOT NULL,   -- ID định danh duy nhất
    filename VARCHAR,                   -- Tên file .wav
    filepath VARCHAR,                   -- Đường dẫn lưu trữ
    transcript TEXT                     -- Nội dung bài nói (từ Whisper)
);

-- Bảng 2: Lưu trữ Vector (Quan hệ 1-1)
CREATE TABLE audio_vectors (
    id             INTEGER PRIMARY KEY
                   REFERENCES audio_metadata(id) ON DELETE CASCADE,
    voice_vector   vector(32),   -- Đặc trưng giọng (đã Z-Score)
    content_vector vector(384)   -- Đặc trưng nội dung (Embedding)
);

-- Index HNSW cho tìm kiếm nhanh
CREATE INDEX ON audio_vectors USING hnsw (voice_vector vector_l2_ops);    -- L2/Euclidean
CREATE INDEX ON audio_vectors USING hnsw (content_vector vector_cosine_ops); -- Cosine
```

### 5.3. Lý do tách 2 bảng (1-1)?

- **Tách biệt metadata và vector**: Khi cần sửa transcript hoặc filepath, không đụng chạm đến dữ liệu vector
- **Hiệu suất truy vấn**: Có thể JOIN chỉ khi cần, tránh load toàn bộ vector 384 chiều khi chỉ cần đọc metadata
- **Mở rộng dễ dàng**: Về sau có thể thêm bảng `audio_vectors_v2` với vector chiều khác mà không phá schema cũ

---

## 6. Sơ đồ Khối Hệ thống

### 6.1. Giai đoạn XÂY DỰNG CSDL (Offline)

```
┌─────────────────┐
│  510 file .wav  │
└────────┬────────┘
         │
    ┌────▼────────────────────────────────┐
    │         feature_extractor.py        │
    │  Pre-emphasis → Frame → Hamming     │
    │  → FFT → Mel → DCT → MFCC          │
    │  + ZCR, Energy, Centroid            │
    │  → 32D Raw Vector / file            │
    └────┬─────────────────────────┬──────┘
         │                         │
    ┌────▼────────────────────┐    │
    │   normalize_features.py │    │
    │   StandardScaler        │    │
    │   (X - μ) / σ           │    │
    │   → 32D Scaled Vector   │    │
    │   → scaler_params.json  │    │
    └────┬────────────────────┘    │
         │                    ┌────▼──────────────────────┐
         │                    │     content_extractor.py   │
         │                    │  Whisper → Transcript       │
         │                    │  all-MiniLM-L6-v2           │
         │                    │  → 384D Embedding / file    │
         │                    └────┬──────────────────────┘
         │                         │
    ┌────▼─────────────────────────▼──────┐
    │         database_loader.py          │
    │   Merge metadata + voice + content  │
    │   INSERT vào PostgreSQL             │
    │   audio_metadata + audio_vectors    │
    └─────────────────────────────────────┘
```

### 6.2. Giai đoạn TÌM KIẾM (Online - Real-time)

```
┌──────────────────────┐
│  File query .wav mới │
└──────────┬───────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │           search_engine.process_query()  │
    │                                          │
    │  NHÁNH 1 (Voice):                        │
    │  extract_features() → 32D Raw            │
    │  Z-Score với scaler_params.json          │
    │  → 32D Scaled Query Vector               │
    │                                          │
    │  NHÁNH 2 (Content):                      │
    │  Whisper.transcribe() → Transcript       │
    │  SentenceTransformer.encode()            │
    │  → 384D Query Embedding                  │
    └──────┬──────────────────┬────────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐
    │ search_voice│    │search_content│
    │             │    │              │
    │ SQL: <->    │    │ SQL: <=>     │
    │ L2 Distance │    │Cosine Dist.  │
    │             │    │              │
    │ Score =     │    │ Score =      │
    │1/(1+d/10)   │    │ 1 - CosDist  │
    └──────┬──────┘    └──────┬───────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼───────┐
    │  Top-3 Voice│    │ Top-3 Content│
    │  (filename, │    │ (filename,   │
    │   score,    │    │  score,      │
    │  transcript)│    │  transcript) │
    └──────┬──────┘    └──────┬───────┘
           │                  │
    ┌──────▼──────────────────▼───────┐
    │          app.py (Streamlit UI)   │
    │   Hiển thị kết quả song song     │
    └──────────────────────────────────┘
```

---

## 7. Kết quả Trung gian Quá trình Tìm kiếm

### 7.1. Ví dụ với file query `_D8YJ6opZYU.wav`

**Bước 1 — Trích xuất Voice Vector (32 chiều, sau Z-Score):**
```
[-0.42, 1.21, -1.05, 0.87, 0.33, -0.91, ..., 0.15, -0.72]  # 32 giá trị
```

**Bước 2 — Transcript (Whisper output):**
```
"the existence of other universes both explain what we see in our
 universe and unambiguously predict what we should see..."
```

**Bước 3 — Content Vector (384 chiều, snippet):**
```
[-0.106, -0.027, 0.063, -0.029, 0.030, 0.044, 0.019, -0.076, ...]
```

**Bước 4 — SQL Voice Query:**
```sql
SELECT m.filename,
       1 / (1 + (v.voice_vector <-> '[−0.42,1.21,...]'::vector) / 10.0) AS score
FROM audio_vectors v JOIN audio_metadata m ON v.id = m.id
ORDER BY score DESC LIMIT 3;
```

**Kết quả Voice:**
| Rank | Filename | Voice Similarity |
|---|---|---|
| #1 | `_D8YJ6opZYU.wav` | 100.00% (chính nó) |
| #2 | `AUL2pMTLIZc.wav` | 68.42% |
| #3 | `BZMeuAibs1A.wav` | 61.15% |

**Kết quả Content:**
| Rank | Filename | Content Similarity |
|---|---|---|
| #1 | `_D8YJ6opZYU.wav` | 96.84% (chính nó, ~3% sai số STT) |
| #2 | `xX0hgcgEX-w.wav` | 78.21% |
| #3 | `pRVchHPpHko.wav` | 71.53% |

> **Lưu ý:** Top-1 Content không đạt 100% vì Whisper transcribe lại file sẽ có sai số nhỏ so với transcript gốc lưu trong DB → vector embedding lệch nhẹ → Cosine < 100%.

---

## 8. Demo & Đánh giá Hệ thống

### 8.1. Chạy ứng dụng
```bash
pip install streamlit openai-whisper sentence-transformers psycopg2-binary numpy scipy
streamlit run app.py
```

### 8.2. Giao diện Demo

| Khu vực | Chức năng |
|---|---|
| **Sidebar** | Thông tin nhóm, quy mô CSDL |
| **Upload** | Kéo thả file `.wav` bất kỳ |
| **Audio Player** | Nghe lại file vừa upload |
| **Search Button** | Kích hoạt toàn bộ pipeline AI |
| **Kết quả Voice** | Top-3 + % Voice Similarity + Audio Player |
| **Kết quả Content** | Top-3 + % Content Similarity + Trích đoạn + Xem đủ Transcript |

### 8.3. Nhận xét Kết quả

**Voice Similarity:**
- Khi query là file trong DB: Top-1 luôn đạt 100%
- Top-2, Top-3 dao động 50-75% sau khi chuẩn hóa Z-Score (cải thiện đáng kể so với 1-2% khi chưa chuẩn hóa)
- Độ chính xác phụ thuộc vào tính đặc trưng của giọng người nói

**Content Similarity:**
- Top-1 với file trong DB: 90-98% (sai số do Whisper non-deterministic)
- Hệ thống tìm được các bài TED Talk cùng chủ đề (vũ trụ, kinh doanh, khoa học...) chính xác cao

### 8.4. Điểm Mạnh & Hạn chế

| | Điểm Mạnh | Hạn chế |
|---|---|---|
| **Voice** | Không cần nhãn giới tính, tự học từ tín hiệu | Nhạy cảm với chất lượng ghi âm, nhiễu nền |
| **Content** | Chịu lỗi chính tả STT, hiểu ngữ nghĩa thực sự | Phụ thuộc vào chất lượng Whisper transcription |
| **Hệ thống** | Hybrid — bổ sung lẫn nhau, HNSW index nhanh | Cần GPU để chạy Whisper nhanh hơn |

---

## 9. Câu hỏi Thường gặp (Q&A)

**Q: Tại sao dùng 13 hệ số MFCC?**
> A: 13 hệ số đầu tiên chứa đủ thông tin âm sắc của giọng nói. Các hệ số cao hơn biểu diễn các biến đổi tần số rất nhanh, thường là nhiễu và không mang thêm thông tin hữu ích. Đây là chuẩn công nghiệp trong ASR từ thập niên 1980.

**Q: Tại sao Top-1 Voice = 100% khi query chính file đó?**
> A: Khoảng cách Euclidean của một vector với chính nó luôn = 0. Lắp vào công thức: `1/(1+0/10) = 1.0 = 100%`.

**Q: Tại sao không dùng Cosine cho Voice thay vì Euclidean?**
> A: Các đặc trưng vật lý như Energy và Centroid mang **ý nghĩa ở độ lớn tuyệt đối** (giọng to/nhỏ, tần số cao/thấp). Cosine bỏ qua độ lớn, chỉ quan tâm phương hướng → sẽ đánh đồng giọng thì thầm với giọng hét to nếu âm sắc giống nhau. Euclidean đo đúng khoảng cách tuyệt đối nên phù hợp hơn.

**Q: File `scaler_params.json` có thể xóa không?**
> A: **Không được xóa.** File này chứa tham số μ và σ để chuẩn hóa vector query mới về cùng hệ quy chiếu với dữ liệu trong DB. Nếu xóa, hệ thống vẫn chạy nhưng kết quả Voice sẽ sai lệch hoàn toàn.

**Q: Tại sao Content Top-1 không đạt 100% dù query chính file đó?**
> A: Mô hình Whisper có tính **non-deterministic** (không tất định) — mỗi lần chạy có thể sinh ra transcript hơi khác (dấu câu, từ luyến âm...). Vector embedding của 2 transcript khác nhau dù rất ít cũng sẽ lệch nhau → Cosine < 100%. Điều này **chứng minh** hệ thống chạy AI thực sự, không "ăn gian".

**Q: Hệ thống HNSW Index là gì?**
> A: HNSW (Hierarchical Navigable Small World) là thuật toán tìm kiếm lân cận gần đúng (ANN) dạng đồ thị. Thay vì so sánh tuần tự O(n), nó duyệt qua các tầng đồ thị để tìm kết quả gần đúng trong O(log n), nhanh hơn hàng ngàn lần với tập dữ liệu lớn.

**Q: Làm sao hệ thống phân biệt được giọng nam/nữ?**
> A: Hệ thống không được cung cấp nhãn giới tính. Tuy nhiên, `centroid_mean` (tần số trọng tâm) của giọng nữ (~3000-4000 Hz) cao hơn đáng kể so với giọng nam (~2000-3000 Hz). Sau chuẩn hóa Z-Score, sự chênh lệch này được bảo tồn và thuật toán Euclidean tự nhiên sẽ xếp các file cùng giới tính gần nhau hơn trong không gian vector.

---

## 10. Cài đặt & Chạy lại từ Đầu

```bash
# 1. Cài thư viện
pip install streamlit openai-whisper sentence-transformers psycopg2-binary numpy scipy pandas

# 2. Trích xuất đặc trưng giọng nói (Giai đoạn 1)
python feature_extractor.py

# 3. Trích xuất đặc trưng nội dung (Giai đoạn 2) — cần GPU/thời gian dài
python content_extractor.py

# 4. Chuẩn hóa Z-Score
python normalize_features.py

# 5. Nạp vào PostgreSQL
python database_loader.py

# 6. Chạy giao diện
streamlit run app.py
```

> ⚠️ **Lưu ý:** Bước 3 cần ~2-3 giờ nếu chạy bằng CPU. Nên chạy trên máy có GPU hoặc dùng Google Colab.
