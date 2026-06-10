# 🎵 Audio Search Engine (CBIR)
**Môn học:** Hệ Cơ sở Dữ liệu Đa phương tiện | **Nhóm:** 05

---

## 1. Tổng quan Dự án

Hệ thống truy vấn âm thanh lai (Audio Retrieval) cho phép người dùng upload một file `.wav` và tìm kiếm ra Top-3 file âm thanh giống nhất trong cơ sở dữ liệu theo **hai tiêu chí song song**:

| Chiều tìm kiếm | Phương pháp | Đặc trưng | Độ đo |
|---|---|---|---|
| **Giọng nói** (Voice) | Xử lý tín hiệu số | Vector 32 chiều (MFCC, ZCR, Energy, Centroid) | Euclidean Distance |
| **Nội dung** (Content) | Deep Learning | Vector 384 chiều (Text Embedding) | Cosine Similarity |

---

## 2. Bộ Dữ liệu

- **Nguồn:** Các bài diễn thuyết TED Talks định dạng `.wav`
- **Quy mô:** 510 file audio
- **Định dạng:** PCM WAV, Mono, 16kHz
- **Nội dung:** Tiếng Anh, dài từ 1 - 3 phút/file, nhiều người nói khác nhau

### Cấu trúc thư mục
```
btl/
├── data/                          # 510 file audio
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

Trong đó:
- $y[n]$: Tín hiệu đầu ra sau khi lọc pre-emphasis tại mẫu thứ $n$
- $x[n]$: Tín hiệu âm thanh gốc (input) tại mẫu thứ $n$
- $x[n-1]$: Mẫu tín hiệu ngay trước đó
- $\alpha = 0.97$: Hệ số pre-emphasis (giá trị chuẩn công nghiệp), quyết định mức độ tăng cường tần số cao

**Bước 2: Framing** — Cắt tín hiệu thành các khung nhỏ chồng lấp:
- Frame size: 25ms (~400 samples tại 16kHz)
- Frame shift: 10ms (~160 samples)

**Bước 3: Hamming Window** — Giảm nhiễu biên khung:

$$w[n] = 0.54 - 0.46 \cdot \cos\left(\frac{2\pi n}{N-1}\right)$$

Trong đó:
- $w[n]$: Giá trị cửa sổ Hamming tại vị trí thứ $n$ trong khung
- $n$: Chỉ số mẫu trong khung, chạy từ $0$ đến $N-1$
- $N$: Tổng số mẫu trong một khung (frame size = 400 mẫu tại 16kHz)
- $0.54$ và $0.46$: Các hằng số của cửa sổ Hamming, được chọn để tối thiểu hóa rò rỉ phổ (spectral leakage) tại biên khung

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

*Công thức 1 — Chuyển đổi tần số Hz sang thang Mel:*

$$\text{Mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

Trong đó:
- $\text{Mel}(f)$: Giá trị tần số trên thang Mel (đơn vị: mel) — thang đo phi tuyến mô phỏng cảm nhận âm thanh của tai người
- $f$: Tần số vật lý đầu vào (đơn vị: Hz)
- $2595$ và $700$: Các hằng số thực nghiệm, đảm bảo thang Mel xấp xỉ đúng cảm nhận thính giác con người
- $\log_{10}$: Logarit cơ số 10 — phản ánh việc tai người cảm nhận tần số theo thang logarit (tần số càng cao, khả năng phân biệt càng kém)

*Công thức 2 — Tính hệ số MFCC bằng biến đổi Cosine rời rạc (DCT):*

$$\text{MFCC}_k = \sum_{m=1}^{M} \log S_m \cdot \cos\left[k\left(m - \frac{1}{2}\right)\frac{\pi}{M}\right]$$

Trong đó:
- $\text{MFCC}_k$: Hệ số MFCC thứ $k$ (hệ thống lấy $k = 1, 2, ..., 13$)
- $k$: Chỉ số hệ số cepstral, quyết định mức độ chi tiết phổ ($k$ nhỏ = đặc trưng thô, $k$ lớn = chi tiết tinh)
- $M$: Tổng số bộ lọc Mel (Mel filterbank), trong hệ thống dùng $M = 40$ bộ lọc
- $m$: Chỉ số bộ lọc Mel, chạy từ $1$ đến $M$
- $S_m$: Năng lượng phổ đầu ra của bộ lọc Mel thứ $m$ (sau khi áp dụng Mel filterbank lên phổ FFT)
- $\log S_m$: Logarit của năng lượng — nén dải động, mô phỏng cảm nhận phi tuyến của tai người
- $\cos[...]$: Phép biến đổi Cosine rời rạc (DCT) — khử tương quan giữa các bộ lọc, nén thông tin vào ít hệ số

**Zero Crossing Rate (ZCR) — Tỷ lệ cắt qua mức 0:**

$$\text{ZCR} = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{1}[x[n] \cdot x[n-1] < 0]$$

Trong đó:
- $\text{ZCR}$: Tỷ lệ số lần tín hiệu đổi dấu (cắt qua trục 0) — giá trị từ $0$ đến $1$
- $N$: Tổng số mẫu tín hiệu trong khung (frame)
- $n$: Chỉ số mẫu, chạy từ $1$ đến $N-1$
- $x[n]$: Biên độ tín hiệu tại mẫu thứ $n$
- $x[n-1]$: Biên độ tín hiệu tại mẫu ngay trước đó
- $\mathbb{1}[\cdot]$: Hàm chỉ thị (indicator function) — bằng $1$ nếu điều kiện bên trong đúng, bằng $0$ nếu sai
- $x[n] \cdot x[n-1] < 0$: Điều kiện hai mẫu liên tiếp **trái dấu** (một dương, một âm) → tín hiệu cắt qua trục 0
- $\frac{1}{N-1}$: Chuẩn hóa bằng tổng số cặp mẫu liền kề, đưa về tỷ lệ phần trăm

**RMS Energy (Root Mean Square Energy) — Năng lượng hiệu dụng:**

$$E = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

Trong đó:
- $E$: Năng lượng RMS của tín hiệu — đo **độ to (loudness)** trung bình của âm thanh trong khung
- $N$: Tổng số mẫu tín hiệu trong khung
- $n$: Chỉ số mẫu, chạy từ $0$ đến $N-1$
- $x[n]$: Biên độ tín hiệu tại mẫu thứ $n$
- $x[n]^2$: Bình phương biên độ — loại bỏ dấu âm/dương, chỉ giữ lại độ lớn
- $\frac{1}{N}$: Lấy trung bình (Mean) trên toàn bộ khung
- $\sqrt{\cdot}$: Căn bậc hai — đưa giá trị về cùng đơn vị với biên độ gốc

**Spectral Centroid — Tần số trọng tâm phổ:**

$$C = \frac{\sum_k f_k \cdot |X[k]|}{\sum_k |X[k]|}$$

Trong đó:
- $C$: Spectral Centroid — tần số "trọng tâm" của phổ (đơn vị: Hz), cho biết vùng tần số mà năng lượng tập trung nhiều nhất. Giọng cao → $C$ lớn, giọng trầm → $C$ nhỏ
- $k$: Chỉ số bin tần số trong phổ FFT
- $f_k$: Tần số vật lý (Hz) tương ứng với bin thứ $k$
- $X[k]$: Hệ số phổ Fourier (FFT) tại bin thứ $k$ — giá trị phức biểu diễn thành phần tần số
- $|X[k]|$: Biên độ phổ (magnitude) tại bin $k$ — phản ánh cường độ năng lượng tại tần số $f_k$
- $\sum_k f_k \cdot |X[k]|$: Tổng có trọng số — mỗi tần số $f_k$ được "cân" bởi năng lượng $|X[k]|$ tại tần số đó
- $\sum_k |X[k]|$: Tổng năng lượng toàn phổ — dùng để chuẩn hóa, đảm bảo $C$ có đơn vị Hz
- Ý nghĩa trực quan: Spectral Centroid là **trung bình có trọng số** (weighted average) của tất cả tần số, với trọng số là năng lượng. Tương tự "trọng tâm" trong vật lý

### 3.4. Tại sao chọn 32 đặc trưng này?

- **MFCC:** Bắt chước cơ chế nghe của tai người (thang Mel phi tuyến). Là đặc trưng tiêu chuẩn trong mọi hệ thống nhận dạng giọng nói (ASR, Speaker ID).
- **ZCR:** Phân biệt âm vô thanh (s, f, sh) với âm hữu thanh (a, e, i). Phản ánh đặc tính ngữ âm của người nói.
- **Energy:** Người nói to/nhỏ khác nhau tạo ra profile năng lượng đặc trưng.
- **Centroid:** Giọng nữ thường có centroid cao hơn giọng nam vì tần số cơ bản (F0) cao hơn.

### 3.5. Chuẩn hóa Z-Score (StandardScaler)

**Vấn đề:** Centroid có giá trị hàng ngàn Hz trong khi MFCC chỉ vài chục → Euclidean bị Centroid "áp đảo".

**Giải pháp:** Chuẩn hóa mỗi cột về Mean=0, Std=1:

$$z = \frac{x - \mu}{\sigma}$$

Trong đó:
- $z$: Giá trị đã chuẩn hóa (standardized value) — không có đơn vị, trung bình = 0, độ lệch chuẩn = 1
- $x$: Giá trị gốc (raw value) của đặc trưng (ví dụ: `centroid_mean = 3500 Hz`)
- $\mu$: Giá trị trung bình (mean) của cột đặc trưng đó trên toàn bộ 510 file trong CSDL
- $\sigma$: Độ lệch chuẩn (standard deviation) của cột đặc trưng đó trên toàn bộ 510 file

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

Trong đó:
- $\vec{A}$: Vector embedding 384 chiều của văn bản thứ nhất (ví dụ: query transcript)
- $\vec{B}$: Vector embedding 384 chiều của văn bản thứ hai (ví dụ: file trong CSDL)
- $\vec{A} \cdot \vec{B}$: Tích vô hướng (dot product) của hai vector = $\sum_{i=1}^{384} A_i \cdot B_i$
- $|\vec{A}|$: Độ dài (norm/magnitude) của vector $\vec{A}$ = $\sqrt{\sum_{i=1}^{384} A_i^2}$
- $|\vec{B}|$: Độ dài (norm/magnitude) của vector $\vec{B}$
- $\theta$: Góc giữa hai vector trong không gian 384 chiều
- $\cos\theta$: Cosine của góc — nằm trong khoảng $[-1, 1]$, giá trị càng gần $1$ thì hai vector càng cùng hướng (nội dung càng giống nhau)

- $\theta = 0°$: Hai văn bản cùng chủ đề hoàn toàn → Score = 100%
- $\theta = 90°$: Hai văn bản không liên quan → Score = 0%

Trong pgvector, toán tử `<=>` trả về **Cosine Distance** (= 1 - Cosine Similarity). Do đó điểm tương đồng hiển thị được tính:
$$\text{Score}_\text{content} = 1 - \text{CosineDistance}(\vec{q}, \vec{d})$$

Trong đó:
- $\text{Score}_\text{content}$: Điểm tương đồng nội dung (Content Similarity Score), giá trị từ $0\%$ đến $100\%$
- $\text{CosineDistance}(\vec{q}, \vec{d})$: Khoảng cách Cosine = $1 - \text{Cosine Similarity}$, do toán tử `<=>` của pgvector trả về
- $\vec{q}$: Vector embedding 384 chiều của file query (file audio người dùng upload)
- $\vec{d}$: Vector embedding 384 chiều của file trong cơ sở dữ liệu

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
| **Hệ thống** | Bổ sung lẫn nhau, HNSW index nhanh | Cần GPU để chạy Whisper nhanh hơn |

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

## 10. CÂU HỎI BẢO VỆ ĐỒ ÁN — Hỏi đáp chi tiết

---

### Câu 1: Mô hình cơ sở dữ liệu là mô hình gì?

**Trả lời:**

Hệ thống sử dụng **mô hình cơ sở dữ liệu quan hệ (Relational Database Model)**, được triển khai trên hệ quản trị cơ sở dữ liệu **PostgreSQL** kết hợp với extension **pgvector**.

Cụ thể:
- **Mô hình quan hệ** tổ chức dữ liệu theo dạng **bảng (table/relation)**, mỗi bảng gồm các **hàng (row/tuple)** đại diện cho một bản ghi và các **cột (column/attribute)** đại diện cho thuộc tính.
- Các bảng được liên kết với nhau thông qua **khóa chính (Primary Key)** và **khóa ngoại (Foreign Key)**, đảm bảo **tính toàn vẹn tham chiếu (Referential Integrity)**.
- PostgreSQL hỗ trợ đầy đủ các tính chất **ACID** (Atomicity, Consistency, Isolation, Durability) — đảm bảo dữ liệu an toàn và nhất quán.
- Extension **pgvector** mở rộng PostgreSQL với kiểu dữ liệu `vector(n)`, cho phép lưu trữ vector đặc trưng đa chiều và thực hiện **tìm kiếm tương đồng (similarity search)** trực tiếp bằng SQL, không cần hệ thống ngoài.

**Tóm lại:** Đây là mô hình quan hệ mở rộng (Extended Relational Model) — giữ nguyên cấu trúc bảng, khóa, ràng buộc của mô hình quan hệ truyền thống, đồng thời bổ sung khả năng xử lý dữ liệu vector đa chiều thông qua pgvector.

---

### Câu 2: Trình bày về dữ liệu, file của em như thế nào? Dùng gì để so sánh các kỹ thuật với nhau?

**Trả lời:**

**Về dữ liệu:**
- Dữ liệu là **510 file âm thanh** định dạng `.wav` (PCM WAV), nội dung là các bài diễn thuyết **TED Talks** bằng tiếng Anh.
- Mỗi file có **1 kênh (Mono)**, tần số lấy mẫu **16,000 Hz (16kHz)**, độ sâu bit **16-bit (int16)**, thời lượng dao động từ **1 đến 3 phút**.
- Tên file được đặt theo **YouTube Video ID** (ví dụ: `GapPidut5nk.wav`), giúp truy nguyên nguồn gốc dễ dàng.
- Ngoài file âm thanh, mỗi file còn có bản phiên âm **transcript** (nội dung văn bản bài nói) đã được lưu sẵn trong `metadata.csv`.

**Về các file CSV trung gian:**

| File | Nội dung | Kích thước |
|---|---|---|
| `metadata.csv` | file_id, filename, filepath, transcript (510 dòng) | ~700KB |
| `audio_voice_features.csv` | 32 đặc trưng vật lý thô (raw) cho mỗi file | ~4.1MB |
| `audio_voice_features_scaled.csv` | 32 đặc trưng đã chuẩn hóa Z-Score | ~330KB |
| `audio_content_features.csv` | Vector 384 chiều text embedding cho mỗi file | ~4.3MB |

**Dùng gì để so sánh các kỹ thuật:**
- Hệ thống có **2 kỹ thuật tìm kiếm song song**: tìm theo **giọng nói (Voice)** và tìm theo **nội dung (Content)**.
- Để so sánh 2 kỹ thuật, ta dùng cùng một file query đầu vào rồi xem kết quả Top-3 của mỗi nhánh:
  - **Voice** dùng **Euclidean Distance** (khoảng cách L2) trên vector 32 chiều → tìm file có giọng nói giống nhau.
  - **Content** dùng **Cosine Similarity** trên vector 384 chiều → tìm file có nội dung chủ đề giống nhau.
- Khi 2 kỹ thuật trả về **các file khác nhau**, điều đó chứng minh chúng đang đo lường **hai khía cạnh khác nhau** (vật lý vs. ngữ nghĩa) → bổ sung lẫn nhau.

---

### Câu 3: Lưu cơ sở dữ liệu bằng cách gì? Đọc file chỗ nào?

**Trả lời:**

**Lưu cơ sở dữ liệu bằng cách nào:**
- CSDL được lưu trong hệ quản trị **PostgreSQL** (phiên bản cài trên localhost, cổng 5432).
- Dữ liệu được nạp vào PostgreSQL bằng script **`database_loader.py`**, quy trình cụ thể:
  1. Đọc 3 file CSV (`metadata.csv`, `audio_voice_features_scaled.csv`, `audio_content_features.csv`).
  2. Merge (nối) dữ liệu dựa trên cột `file_id` chung.
  3. Kết nối PostgreSQL bằng thư viện `psycopg2`.
  4. Tạo 2 bảng: `audio_metadata` và `audio_vectors`.
  5. Chèn (INSERT) từng bản ghi vào 2 bảng bằng SQL.
  6. Xây dựng chỉ mục **HNSW** trên 2 cột vector để tăng tốc truy vấn.
  7. Commit giao dịch, đóng kết nối.

**Đọc file ở chỗ nào (trong code):**

| File code | Hàm đọc file | Đọc file gì |
|---|---|---|
| `feature_extractor.py` dòng 50 | `wavfile.read(filepath)` — thư viện `scipy.io` | Đọc file `.wav` để trích xuất tín hiệu âm thanh (signal) và tần số lấy mẫu (sample_rate) |
| `feature_extractor.py` dòng 181 | `pd.read_csv(metadata_path)` | Đọc `metadata.csv` để lấy danh sách file cần xử lý |
| `content_extractor.py` dòng 35 | `pd.read_csv(input_csv)` | Đọc `metadata.csv` để lấy transcript và encode thành vector |
| `normalize_features.py` dòng 6 | `pd.read_csv('audio_voice_features.csv')` | Đọc đặc trưng raw để chuẩn hóa Z-Score |
| `database_loader.py` dòng 12-28 | `pd.read_csv(...)` (3 lần) | Đọc cả 3 file CSV để merge và nạp vào DB |
| `search_engine.py` dòng 26 | `open('scaler_params.json')` + `json.load()` | Đọc tham số μ, σ để chuẩn hóa query mới |

---

### Câu 4: Thầy đưa em 2 file, so sánh đặc điểm 1 cách cảm tính bằng cách gì, làm sao để so sánh cái đấy theo cảm tính, cảm tính giống nhau cái gì và khác nhau cái gì? Nghe? Còn cách nào khác không?

**Trả lời:**

**Cách so sánh cảm tính 2 file âm thanh:**

**Cách 1 — Nghe bằng tai:**
- Mở 2 file và **nghe lần lượt** (hoặc nghe song song nếu có 2 loa). Khi nghe, tai người có thể cảm nhận được:
  - **Giống nhau về giọng nói:** Cùng giới tính (nam/nữ), cùng tông giọng (trầm/cao), cùng tốc độ nói (nhanh/chậm), cùng ngữ điệu (đều đều hay lên xuống).
  - **Giống nhau về nội dung:** Cùng nói về một chủ đề (ví dụ: cả 2 đều nói về khoa học, hoặc đều nói về giáo dục).
  - **Khác nhau:** Giọng nam vs. giọng nữ, tốc độ nhanh vs. chậm, chủ đề hoàn toàn khác nhau, có nhạc nền vs. không có.
- Đây là cách trực quan nhất nhưng **chủ quan và không lượng hóa được** — hai người nghe có thể cho kết quả khác nhau.

**Cách 2 — Nhìn bằng mắt qua trực quan hóa (Visualization):**
- Ngoài nghe, ta còn có thể **vẽ Waveform (dạng sóng)** và **Spectrogram (phổ tần số-thời gian)** của 2 file rồi **so sánh bằng mắt**:
  - **Waveform:** Quan sát biên độ sóng — file nào nói to hơn sẽ có biên độ lớn hơn, file nào có nhiều khoảng im lặng sẽ có đoạn sóng bằng phẳng.
  - **Spectrogram:** Hiển thị phân bố năng lượng theo tần số và thời gian. Giọng nam sẽ có năng lượng tập trung ở **vùng tần số thấp** (dưới 3000 Hz), giọng nữ sẽ phân bố lên **vùng tần số cao hơn**. Hai file cùng người nói sẽ có pattern spectrogram tương tự nhau.
  - Trong code, em đã viết hàm `plot_check()` trong `feature_extractor.py` (dòng 148-177) để vẽ cả Waveform và Spectrogram cho bất kỳ file nào.

**Cách 3 — Đọc transcript (nội dung bài nói):**
- Nếu đã có transcript trong `metadata.csv`, ta có thể **đọc nội dung văn bản** để xác định 2 file nói về chủ đề gì, có giống nhau không. Đây là so sánh cảm tính về **ngữ nghĩa**.

**Tại sao cần chuyển sang so sánh định lượng:**
- Phương pháp cảm tính chỉ khả thi khi có 2 file. Với **510 file trong CSDL**, không thể nghe hết để tìm file giống nhất → cần **trích xuất đặc trưng số học** và dùng **thuật toán so sánh tự động**.

---

### Câu 5: Bảng này tên là gì và bảng kia có tên là gì? Khóa chính bảng này là gì, bảng kia khóa chính là gì? Title là khóa chính, trùng nhau à?

**Trả lời:**

**Tên 2 bảng trong CSDL:**

| # | Tên bảng | Chức năng |
|---|---|---|
| Bảng 1 | **`audio_metadata`** | Lưu thông tin mô tả (metadata) của từng file audio |
| Bảng 2 | **`audio_vectors`** | Lưu các vector đặc trưng (voice + content) của từng file audio |

**Khóa chính:**
- Bảng `audio_metadata`: Khóa chính là **`id`** (kiểu `SERIAL` — số nguyên tự tăng: 1, 2, 3, ..., 510). Ngoài ra cột `file_id` có ràng buộc `UNIQUE NOT NULL` nên cũng đóng vai trò như một **khóa ứng viên (Candidate Key)**.
- Bảng `audio_vectors`: Khóa chính cũng là **`id`** (kiểu `INTEGER`) — đồng thời là **khóa ngoại (Foreign Key)** tham chiếu đến `audio_metadata(id)` với ràng buộc `ON DELETE CASCADE`.

**Hai bảng có cùng khóa chính không?**
- **Có.** Cả 2 bảng đều dùng cột `id` làm khóa chính, và giá trị `id` ở `audio_vectors` phải **trùng khớp** với `id` tương ứng bên `audio_metadata`. Đây là thiết kế **quan hệ 1-1 (One-to-One)** — mỗi bản ghi trong `audio_metadata` có **đúng 1** bản ghi tương ứng trong `audio_vectors`, và ngược lại.

**Title có phải khóa chính không? Có trùng không?**
- Trong hệ thống này **không có cột `title`**. Cột gần nhất có thể nhầm là `filename` (ví dụ: `GapPidut5nk.wav`). `filename` **không phải khóa chính** — nó chỉ là thuộc tính mô tả.
- Khóa chính là `id` (số nguyên tự tăng) — **không bao giờ trùng nhau** vì PostgreSQL tự sinh giá trị duy nhất cho mỗi bản ghi.
- Cột `file_id` (ví dụ: `GapPidut5nk`) cũng có ràng buộc `UNIQUE` nên cũng không trùng nhau.

---

### Câu 6: Phần đọc file của em đâu? Sau khi đọc xong chúng ta có cái gì, lưu vào đâu? File CSV có chứa đặc trưng không? Đặc trưng đó như thế nào?

**Trả lời:**

**Phần đọc file nằm ở đâu trong code:**

1. **Đọc file WAV (tín hiệu âm thanh):**
   - File: `feature_extractor.py`, hàm `extract_features()`, **dòng 50**:
     ```python
     sample_rate, signal = wavfile.read(filepath)
     ```
   - Dùng thư viện `scipy.io.wavfile.read()` — đọc trực tiếp file `.wav` nhị phân, trả về:
     - `sample_rate`: Tần số lấy mẫu (16000 Hz)
     - `signal`: Mảng numpy chứa biên độ từng mẫu (dạng `int16`, kích thước ví dụ `(1654443,)` cho file ~103 giây)

2. **Đọc file CSV (metadata, đặc trưng):**
   - `feature_extractor.py` dòng 181: `pd.read_csv('metadata.csv')` — lấy danh sách filepath
   - `content_extractor.py` dòng 35: `pd.read_csv('metadata.csv')` — lấy transcript
   - `normalize_features.py` dòng 6: `pd.read_csv('audio_voice_features.csv')` — đọc đặc trưng thô
   - `database_loader.py` dòng 12-28: đọc cả 3 file CSV

**Sau khi đọc xong có cái gì, lưu vào đâu:**

| Bước | Đọc cái gì | Sau xử lý có cái gì | Lưu vào đâu |
|---|---|---|---|
| Giai đoạn 1 | 510 file `.wav` | 510 vector 32 chiều (MFCC, ZCR, Energy, Centroid) | `audio_voice_features.csv` |
| Chuẩn hóa | `audio_voice_features.csv` | 510 vector 32 chiều đã Z-Score + tham số μ,σ | `audio_voice_features_scaled.csv` + `scaler_params.json` |
| Giai đoạn 2 | 510 transcript từ `metadata.csv` | 510 vector 384 chiều (text embedding) | `audio_content_features.csv` |
| Nạp DB | Cả 3 file CSV | Merge lại thành 510 bản ghi hoàn chỉnh | PostgreSQL (2 bảng: `audio_metadata` + `audio_vectors`) |

**File CSV có chứa đặc trưng không?**
- **Có.** Cả `audio_voice_features.csv` và `audio_content_features.csv` đều chứa đặc trưng.

**Đặc trưng đó như thế nào?**

- **`audio_voice_features.csv`:** Mỗi dòng là 1 file, gồm 2 cột ID (`file_id`, `filename`) + **32 cột số thực** (ví dụ: `mfcc_1_mean = 285.80`, `mfcc_1_std = 74.10`, ..., `centroid_std = 999.03`). Đây là đặc trưng **vật lý/âm học** trích xuất bằng xử lý tín hiệu số (không dùng AI).

- **`audio_content_features.csv`:** Mỗi dòng là 1 file, gồm `file_id`, `filename`, và 1 cột `content_vector` chứa **chuỗi JSON là mảng 384 số thực** (ví dụ: `[0.053, -0.004, 0.009, ..., -0.052]`). Đây là đặc trưng **ngữ nghĩa** được tạo bởi model AI Sentence-Transformers.

---

### Câu 7: Phần đọc cơ sở dữ liệu để tìm kiếm đâu?

**Trả lời:**

Phần đọc CSDL để tìm kiếm nằm trong file **`search_engine.py`**, cụ thể là 2 hàm:

**1. Hàm `search_voice()` (dòng 82-107) — Tìm kiếm theo giọng nói:**
```python
def search_voice(self, voice_vector, top_k=3):
    conn = psycopg2.connect(**self.db_params)  # Kết nối PostgreSQL
    cur = conn.cursor()
    
    query = """
        SELECT m.filename, 
               1 / (1 + (v.voice_vector <-> %s::vector) / 10.0) AS similarity_score,
               m.transcript
        FROM audio_vectors v
        JOIN audio_metadata m ON v.id = m.id
        ORDER BY similarity_score DESC
        LIMIT %s;
    """
    cur.execute(query, (json.dumps(voice_vector), top_k))
    results = cur.fetchall()
    # ...
```
- Kết nối PostgreSQL bằng `psycopg2.connect()`.
- Thực hiện câu lệnh SQL sử dụng toán tử `<->` (L2 Distance) của pgvector.
- JOIN 2 bảng `audio_vectors` và `audio_metadata` trên cột `id`.
- Sắp xếp theo `similarity_score DESC` và lấy Top-3 (`LIMIT 3`).

**2. Hàm `search_content()` (dòng 109-133) — Tìm kiếm theo nội dung:**
```python
def search_content(self, content_vector, top_k=3):
    conn = psycopg2.connect(**self.db_params)
    cur = conn.cursor()
    
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
    # ...
```
- Tương tự, nhưng dùng toán tử `<=>` (Cosine Distance) cho content_vector.

**Nơi gọi 2 hàm này từ giao diện:**
- File `app.py`, dòng 79-80:
  ```python
  top_voices = engine.search_voice(voice_vec, top_k=3)
  top_contents = engine.search_content(content_vec, top_k=3)
  ```

---

### Câu 8: Có bao nhiêu file tất cả? Chúng ta có 2 file âm thanh, làm thế nào để ta cảm nhận hoặc thấy giống nhau?

**Trả lời:**

**Có bao nhiêu file:**
- **Tổng cộng 510 file** âm thanh `.wav` nằm trong thư mục `data/`.
- 510 files trong thư mục data

**Làm thế nào để cảm nhận/thấy 2 file âm thanh giống nhau:**

**Cách 1 — Nghe bằng tai (cảm nhận thính giác):**
- Phát 2 file lần lượt và nghe. Tai người có thể cảm nhận:
  - **Giọng nói giống nhau:** cùng giới tính, cùng tông giọng, cùng nhịp nói, cùng chất giọng (khàn, trong, trẻ, già...).
  - **Nội dung giống nhau:** cùng chủ đề, dùng từ ngữ tương tự.
  - **Cảm xúc giống nhau:** cùng hào hứng, cùng nghiêm túc, cùng buồn.

**Cách 2 — Nhìn bằng mắt (trực quan hóa):**
- Vẽ **Waveform** (biểu đồ biên độ-thời gian): So sánh hình dáng sóng — file nào nói to hơn, file nào có nhiều đoạn im lặng.
- Vẽ **Spectrogram** (biểu đồ tần số-thời gian-cường độ): So sánh vùng tần số — giọng trầm tập trung vùng thấp, giọng cao tập trung vùng cao.
- Code hỗ trợ: hàm `plot_check()` trong `feature_extractor.py` (dòng 148-177).

**Cách 3 — So sánh bằng số liệu (định lượng - cách hệ thống dùng):**
- Trích xuất **đặc trưng số học** (vector 32 chiều cho giọng nói, vector 384 chiều cho nội dung) rồi tính **khoảng cách/độ tương đồng** bằng công thức toán học.
- Đây chính là cách mà hệ thống tự động hóa: thay vì con người phải nghe 510 file, hệ thống tính toán và trả về Top-3 giống nhất chỉ trong vài giây.

---

### Câu 9: Cơ sở dữ liệu có mấy bảng? Bảng thứ nhất là gì, thứ hai là gì? Hai bảng này có liên kết với nhau không, có cùng khóa chính không? Tại sao không làm 1 bảng mà lại tách thành 2 bảng?

**Trả lời:**

**CSDL có mấy bảng:**
- Có **2 bảng** chính.

**Bảng thứ nhất — `audio_metadata`:**

| Cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `id` | SERIAL (auto-increment) | **Khóa chính** — số nguyên tự tăng (1, 2, ..., 510) |
| `file_id` | VARCHAR (UNIQUE) | ID định danh duy nhất (ví dụ: `GapPidut5nk`) |
| `filename` | VARCHAR | Tên file `.wav` (ví dụ: `GapPidut5nk.wav`) |
| `filepath` | VARCHAR | Đường dẫn lưu trữ (ví dụ: `data/GapPidut5nk.wav`) |
| `transcript` | TEXT | Nội dung phiên âm bài nói (từ Whisper) |

**Bảng thứ hai — `audio_vectors`:**

| Cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `id` | INTEGER | **Khóa chính + Khóa ngoại** → tham chiếu `audio_metadata(id)` |
| `voice_vector` | vector(32) | Vector đặc trưng giọng nói 32 chiều (đã Z-Score) |
| `content_vector` | vector(384) | Vector đặc trưng nội dung 384 chiều (Text Embedding) |

**Hai bảng có liên kết không:**
- **Có.** Liên kết thông qua cột `id` — bảng `audio_vectors` có `id` là khóa ngoại tham chiếu đến `audio_metadata(id)` với ràng buộc `ON DELETE CASCADE` (xóa metadata thì tự động xóa vector tương ứng).
- Quan hệ: **1-1 (One-to-One)** — mỗi bản ghi metadata tương ứng đúng 1 bản ghi vector.

**Có cùng khóa chính không:**
- **Cùng cột khóa chính `id`**, cùng giá trị. Ví dụ: file `GapPidut5nk` có `id = 1` ở cả 2 bảng.

**Tại sao không làm 1 bảng mà tách thành 2 bảng?**

Có **3 lý do chính:**

1. **Tách biệt dữ liệu nhẹ và nặng (Performance):**
   - `audio_metadata` chỉ chứa text nhẹ (~1KB/bản ghi). Khi muốn tra cứu thông tin file (tên, transcript), chỉ cần đọc bảng này mà **không phải load 2 vector lớn** (32 + 384 = 416 số thực).
   - `audio_vectors` chứa dữ liệu nặng (vector chiều cao). Khi tìm kiếm vector, hệ thống chỉ cần truy cập bảng này, rồi JOIN lấy thông tin metadata khi cần hiển thị.
   - Nếu gộp 1 bảng, **mỗi lần đọc metadata sẽ phải load kèm 416 số thực** → lãng phí bộ nhớ và chậm.

2. **Dễ bảo trì và mở rộng (Maintainability):**
   - Khi cần sửa transcript hoặc filepath → chỉ UPDATE bảng `audio_metadata`, **không đụng chạm đến vector**.
   - Khi muốn thêm loại vector mới (ví dụ: vector emotion, vector rhythm) → chỉ thêm cột vào `audio_vectors` hoặc tạo bảng vector mới, **không ảnh hưởng metadata**.

3. **Nguyên tắc chuẩn hóa CSDL (Normalization):**
   - Metadata (thông tin mô tả) và Vector (dữ liệu tính toán) thuộc **hai nhóm ngữ nghĩa khác nhau**. Tách riêng tuân thủ nguyên tắc **Single Responsibility** trong thiết kế CSDL — mỗi bảng chỉ phụ trách 1 loại thông tin.

---

### Câu 10: Trình bày cho thầy về các thuộc tính? Ý nghĩa thông tin của từng đặc trưng là gì?

**Trả lời:**

**Tổng quan về các thuộc tính:**
- Hệ thống trích xuất **32 thuộc tính (đặc trưng) vật lý** cho mỗi file audio để biểu diễn **đặc tính giọng nói (Voice)**.
- Ngoài ra còn có **384 thuộc tính ngữ nghĩa** cho **đặc tính nội dung (Content)**.
- Mỗi thuộc tính vật lý có 2 giá trị: **Mean** (giá trị trung bình qua tất cả các frame) và **Std** (độ lệch chuẩn — mức độ biến động qua các frame).

---

#### A. NHÓM ĐẶC TRƯNG MFCC (26 thuộc tính: 13 hệ số × 2 Mean/Std)

MFCC (Mel-Frequency Cepstral Coefficients) là đặc trưng mô phỏng cơ chế nghe của tai người trên thang Mel. Được tính qua quy trình: Pre-emphasis → Framing (25ms) → Hamming Window → FFT (512 điểm) → 40 Mel Filterbank → Log → DCT → lấy 13 hệ số đầu.

| STT | Tên thuộc tính | Ý nghĩa cụ thể |
|---|---|---|
| 1 | `mfcc_1_mean` | **Năng lượng tổng thể của phổ Mel (trung bình).** Còn gọi là "Energy Coefficient". Phản ánh **độ to trung bình** của giọng nói. Giá trị cao = nói to, rõ ràng. Giá trị thấp = nói nhỏ hoặc nhiều đoạn im lặng. Ví dụ: file `GapPidut5nk.wav` có `mfcc_1_mean = 285.80`, trung bình 510 file là `μ = 272.63`. |
| 2 | `mfcc_1_std` | **Độ biến động năng lượng phổ Mel.** Std cao = năng lượng thay đổi nhiều giữa các frame (có lúc to lúc nhỏ) → giọng nói biểu cảm. Std thấp = năng lượng đều đều → giọng monotone. |
| 3 | `mfcc_2_mean` | **Cân bằng năng lượng giữa tần số thấp và cao (trung bình).** Phản ánh sự phân bố phổ — giọng **trầm** (năng lượng tập trung tần số thấp) sẽ có giá trị khác giọng **cao** (năng lượng ở tần số cao). Đây là đặc trưng quan trọng nhất để **phân biệt giọng nam/nữ** sau MFCC₁. |
| 4 | `mfcc_2_std` | **Độ biến động cân bằng phổ.** Std cao = phổ thay đổi nhiều giữa các đoạn nói (ví dụ: người nói chuyển giọng giữa các câu). |
| 5 | `mfcc_3_mean` | **Hình dạng phổ bậc 3 (trung bình).** Bắt đầu mô tả **chi tiết hình bao phổ (spectral envelope)** — đường cong bao quanh các đỉnh phổ. Mỗi người nói có hình bao phổ đặc trưng do cấu tạo thanh quản, khoang miệng, khoang mũi khác nhau. |
| 6 | `mfcc_3_std` | **Độ biến động hình dạng phổ bậc 3.** |
| 7-8 | `mfcc_4_mean/std` | **Chi tiết phổ bậc 4.** Mô tả các **formant** (tần số cộng hưởng) của đường thanh âm — ảnh hưởng bởi hình dạng miệng, lưỡi khi phát âm. |
| 9-10 | `mfcc_5_mean/std` | **Chi tiết phổ bậc 5.** Tiếp tục phân tích cấu trúc formant chi tiết hơn. |
| 11-12 | `mfcc_6_mean/std` | **Chi tiết phổ bậc 6.** |
| 13-14 | `mfcc_7_mean/std` | **Chi tiết phổ bậc 7.** Từ MFCC₅ trở đi, các hệ số mô tả **các biến đổi nhanh hơn** của phổ — biểu diễn texture (kết cấu) tinh tế của giọng nói như độ khàn, rè, mũi. |
| 15-16 | `mfcc_8_mean/std` | **Chi tiết phổ bậc 8.** |
| 17-18 | `mfcc_9_mean/std` | **Chi tiết phổ bậc 9.** |
| 19-20 | `mfcc_10_mean/std` | **Chi tiết phổ bậc 10.** |
| 21-22 | `mfcc_11_mean/std` | **Chi tiết phổ bậc 11.** Các hệ số MFCC bậc cao (9-13) mang thông tin **rất chi tiết** về đặc trưng cá nhân của từng người nói, giúp phân biệt 2 người có cùng giới tính, cùng tông giọng. |
| 23-24 | `mfcc_12_mean/std` | **Chi tiết phổ bậc 12.** |
| 25-26 | `mfcc_13_mean/std` | **Chi tiết phổ bậc 13.** Hệ số cuối cùng — thông tin phổ chi tiết nhất được giữ lại. Các hệ số từ 14 trở đi bị loại vì chủ yếu là nhiễu, không mang thêm thông tin hữu ích. Đây là chuẩn công nghiệp từ năm 1980 (Davis & Mermelstein). |

**Tóm tắt vai trò MFCC:**
- **MFCC₁:** Năng lượng tổng thể (to/nhỏ)
- **MFCC₂:** Phân bố phổ (trầm/cao, nam/nữ)
- **MFCC₃-₅:** Hình bao phổ thô — cấu trúc formant cơ bản (đặc điểm thanh quản)
- **MFCC₆-₉:** Texture giọng nói (khàn, trong, mũi, rè...)
- **MFCC₁₀-₁₃:** Chi tiết cá nhân — "dấu vân tay âm thanh" để phân biệt từng người

---

#### B. NHÓM ĐẶC TRƯNG ZCR (2 thuộc tính)

ZCR (Zero Crossing Rate) = Tỷ lệ đổi dấu của tín hiệu trong mỗi frame. Công thức: đếm số lần tín hiệu đổi từ dương sang âm (hoặc ngược lại) rồi chia cho tổng số mẫu.

| STT | Tên thuộc tính | Ý nghĩa cụ thể |
|---|---|---|
| 27 | `zcr_mean` | **Tỷ lệ cắt qua 0 trung bình.** Phản ánh **tính chất ngữ âm** của giọng nói. ZCR **cao** → nhiều âm **vô thanh** (consonant: "s", "f", "sh", "t") — giọng nói có nhiều phụ âm xát, xì hơi. ZCR **thấp** → nhiều âm **hữu thanh** (vowel: "a", "e", "i", "o", "u") — giọng nói mượt, ít phụ âm. Ví dụ: tiếng Anh Mỹ thường có ZCR cao hơn tiếng Anh Ấn Độ vì phát âm phụ âm rõ hơn. Trung bình 510 file: `μ = 0.314`, `σ = 0.039`. |
| 28 | `zcr_std` | **Độ biến động ZCR.** Std cao = tín hiệu **xen kẽ nhiều giữa âm hữu thanh và vô thanh** → giọng nói tự nhiên, có nhịp. Std thấp = giọng nói đều đều (ví dụ: đọc sách vs. trò chuyện). |

---

#### C. NHÓM ĐẶC TRƯNG ENERGY (2 thuộc tính)

RMS Energy = Căn bậc hai trung bình bình phương biên độ. Đo **năng lượng vật lý thực tế** (loudness) của tín hiệu trong mỗi frame.

| STT | Tên thuộc tính | Ý nghĩa cụ thể |
|---|---|---|
| 29 | `energy_mean` | **Năng lượng trung bình (RMS).** Đo **độ to tuyệt đối** của file audio. Giá trị cao = micro gần, người nói to, tín hiệu mạnh. Giá trị thấp = micro xa, người nói nhỏ, hoặc nhiều đoạn im lặng. Khác với `mfcc_1_mean` ở chỗ: MFCC₁ đo năng lượng **trên thang Mel** (mô phỏng tai người), còn `energy_mean` đo năng lượng **vật lý thuần túy** (biên độ sóng). Trung bình 510 file: `μ = 500.29`, `σ = 116.42`. |
| 30 | `energy_std` | **Độ biến động năng lượng.** Std cao = có đoạn nói to đoạn nói nhỏ → giọng nói **biểu cảm**, nhấn nhá. Std thấp = âm lượng đều → giọng nói **monotone** hoặc file đã được xử lý chuẩn hóa âm lượng (AGC). |

---

#### D. NHÓM ĐẶC TRƯNG SPECTRAL CENTROID (2 thuộc tính)

Spectral Centroid = Tần số trọng tâm của phổ. Tính bằng trung bình có trọng số của tần số, với trọng số là biên độ phổ tại từng tần số. Đơn vị: Hz.

| STT | Tên thuộc tính | Ý nghĩa cụ thể |
|---|---|---|
| 31 | `centroid_mean` | **Tần số trọng tâm phổ trung bình (Hz).** Thể hiện **"độ sáng" (brightness)** của âm thanh — giọng nghe "sáng" hay "tối". Centroid **cao** (~3000-4000 Hz) → **giọng nữ**, giọng trẻ em, hoặc giọng nói có nhiều âm sắc cao. Centroid **thấp** (~2000-3000 Hz) → **giọng nam**, giọng trầm. Đây là đặc trưng **mạnh nhất để phân biệt giới tính** người nói mà không cần nhãn. Trung bình 510 file: `μ = 3028.08 Hz`, `σ = 276.40 Hz`. |
| 32 | `centroid_std` | **Độ biến động tần số trọng tâm.** Std cao = phổ thay đổi nhiều giữa các frame → người nói có **biểu cảm phong phú**, thay đổi cao độ nhiều (ví dụ: diễn giả hào hứng). Std thấp = phổ ổn định → giọng nói đều, ít biểu cảm. Trung bình: `μ = 1178.38 Hz`, `σ = 150.26 Hz`. |

---

#### E. ĐẶC TRƯNG NỘI DUNG (Content Feature — 384 thuộc tính)

Ngoài 32 đặc trưng vật lý, mỗi file còn có **1 vector 384 chiều** biểu diễn **ngữ nghĩa nội dung** bài nói:

| Thuộc tính | Ý nghĩa |
|---|---|
| `content_vector` (384 chiều) | **Text Embedding** — vector đại diện cho ý nghĩa ngữ nghĩa toàn bộ bài nói. Được tạo bởi model **all-MiniLM-L6-v2** (Sentence-Transformers, BERT-based). Mỗi chiều là 1 giá trị số thực (ví dụ: `0.053, -0.004, ...`), không có ý nghĩa riêng lẻ — chỉ khi **kết hợp cả 384 chiều** mới biểu diễn được vị trí của bài nói trong không gian ngữ nghĩa. Hai bài nói cùng chủ đề sẽ có vector chỉ về cùng hướng → Cosine Similarity cao. |

**Tổng kết: Mỗi file audio có tổng cộng 32 + 384 = 416 thuộc tính số học** để biểu diễn cả đặc tính giọng nói lẫn nội dung ngữ nghĩa.

---

### Câu 11: Bảng thứ nhất lưu thuộc tính gì, bảng thứ 2 lưu gì? Bảng thứ nhất có bao nhiêu bản ghi, bảng thứ 2 có bao nhiêu bản ghi?

**Trả lời:**

**Bảng thứ nhất — `audio_metadata` — Lưu thuộc tính gì:**
- Lưu **thông tin mô tả (metadata)** của từng file audio:
  - `id`: Số thứ tự tự tăng (khóa chính)
  - `file_id`: Mã định danh duy nhất (YouTube Video ID)
  - `filename`: Tên file `.wav`
  - `filepath`: Đường dẫn lưu trữ trên ổ đĩa
  - `transcript`: Nội dung phiên âm toàn bộ bài nói (dạng TEXT, có thể dài hàng trăm từ)
- → Bảng này chứa **dữ liệu text** thuần túy, không chứa vector số học.

**Bảng thứ hai — `audio_vectors` — Lưu thuộc tính gì:**
- Lưu **vector đặc trưng số học** của từng file audio:
  - `id`: Khóa chính + Khóa ngoại (tham chiếu `audio_metadata`)
  - `voice_vector`: Vector **32 chiều** (kiểu `vector(32)`) — chứa 32 đặc trưng giọng nói (MFCC, ZCR, Energy, Centroid) đã được chuẩn hóa Z-Score
  - `content_vector`: Vector **384 chiều** (kiểu `vector(384)`) — chứa 384 giá trị text embedding từ model all-MiniLM-L6-v2
- → Bảng này chứa **dữ liệu vector dạng số** dùng cho tính toán tương đồng.

**Số bản ghi:**

| Bảng | Số bản ghi | Lý do |
|---|---|---|
| `audio_metadata` | **510** bản ghi | Tương ứng 510 file audio trong thư mục `data/` |
| `audio_vectors` | **510** bản ghi | Quan hệ 1-1 với `audio_metadata`, mỗi file có đúng 1 vector |

Cả 2 bảng có **cùng số bản ghi (510)** vì quan hệ 1-1 — mỗi file audio có đúng 1 dòng metadata và đúng 1 dòng vector. Điều này được đảm bảo bởi khóa ngoại `REFERENCES audio_metadata(id)`.

---

### Câu 12: Thuộc tính energy tính như thế nào?

**Trả lời:**

Thuộc tính energy được tính bằng công thức **RMS Energy (Root Mean Square Energy)** — căn bậc hai của trung bình bình phương biên độ.

**Công thức toán học:**

$$E_{frame} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

Trong đó:
- $x[n]$: Giá trị biên độ mẫu thứ $n$ trong frame (sau khi đã qua Pre-emphasis + Hamming Window)
- $N$: Số mẫu trong 1 frame (= frame_length = 400 mẫu tại 16kHz với frame size 25ms)
- $E_{frame}$: Giá trị RMS Energy của 1 frame

**Quy trình cụ thể:**

1. **Bước 1 — Tính Energy cho từng frame:**
   - File audio được chia thành nhiều frame (mỗi frame 25ms, bước nhảy 10ms).
   - Với mỗi frame, tính bình phương từng mẫu → lấy trung bình → lấy căn bậc hai.
   - Mã nguồn trong `feature_extractor.py`, dòng 123:
     ```python
     rms_energy = np.sqrt(np.mean(frames**2, axis=1))
     ```

2. **Bước 2 — Tính Mean và Std trên toàn bộ frame:**
   - Sau khi có mảng RMS Energy của tất cả frame, tính:
     - `energy_mean = np.mean(rms_energy)` — **năng lượng trung bình** của cả file (dòng 140)
     - `energy_std = np.std(rms_energy)` — **độ biến động năng lượng** (dòng 141)

**Ý nghĩa:**
- `energy_mean` **cao** → người nói **nói to**, micro gần, tín hiệu mạnh.
- `energy_mean` **thấp** → người nói **nói nhỏ**, micro xa, hoặc nhiều đoạn im lặng.
- `energy_std` **cao** → năng lượng **thay đổi nhiều** (có đoạn to đoạn nhỏ) → giọng nói biểu cảm.
- `energy_std` **thấp** → năng lượng **đều đều** → giọng nói monotone.

**Giá trị ví dụ:** File `GapPidut5nk.wav` có `energy_mean = 427.10`, `energy_std = 491.12`. Giá trị trung bình toàn bộ 510 file: `μ = 500.29`, `σ = 116.42` (từ `scaler_params.json`).

**Tại sao dùng RMS thay vì tổng bình phương?**
- RMS cho giá trị đại diện **trung bình hiệu dụng** của tín hiệu, không bị ảnh hưởng bởi độ dài frame. Nó tỷ lệ thuận với **biên độ cảm nhận** (perceived loudness) của tai người.

---

### Câu 13: Có 2 file âm thanh thì so sánh nội dung của nó bằng cách gì? Nghe thì xác định được các thông tin gì và đặc trưng gì? Ngoài nghe ra dùng gì nữa? Nếu không có cách ta cảm nhận thì làm sao xác định được thuộc tính phù hợp?

**Trả lời:**

**So sánh nội dung 2 file âm thanh bằng cách gì:**

**Cách 1 — Nghe:**
Khi nghe, tai người có thể xác định được các thông tin và đặc trưng sau:
- **Về giọng nói (đặc trưng vật lý):**
  - Giọng nam hay nữ (liên quan đến `centroid_mean` — tần số trọng tâm)
  - Giọng trầm hay cao (liên quan đến MFCC, đặc biệt là MFCC₂)
  - Nói to hay nhỏ (liên quan đến `energy_mean`)
  - Nói nhanh hay chậm (liên quan đến `zcr_mean` — tỷ lệ cắt qua 0)
  - Giọng khàn, trong, mũi... (liên quan đến MFCC₃-₁₃ — hình bao phổ)
- **Về nội dung (ngữ nghĩa):**
  - Chủ đề bài nói là gì (khoa học, nghệ thuật, kinh doanh...)
  - Từ khóa lặp lại nhiều
  - Cảm xúc và giọng điệu (hào hứng, buồn, nghiêm túc)

**Cách 2 — Nhìn (trực quan hóa):**
- **Waveform:** Quan sát biên độ sóng → biết được file nào to hơn, đoạn nào im lặng.
- **Spectrogram:** Quan sát phân bố tần số → biết giọng trầm/cao, có nhiễu không.
- Code hỗ trợ: `plot_check()` trong `feature_extractor.py`.

**Cách 3 — Đọc transcript:**
- Dùng AI (Whisper) chuyển âm thanh thành văn bản, rồi đọc và so sánh nội dung 2 bài nói.

**Nếu không có cách cảm nhận (không nghe, không nhìn), làm sao xác định thuộc tính phù hợp:**

Đây chính là bài toán cốt lõi của xử lý tín hiệu số. Câu trả lời là **dựa vào lý thuyết xử lý tín hiệu và nghiên cứu khoa học:**

1. **Từ lý thuyết vật lý âm thanh:**
   - Âm thanh là sóng → có **tần số, biên độ, phổ** → đây là cơ sở để chọn các đặc trưng: Energy (biên độ), Spectral Centroid (tần số trọng tâm), ZCR (tính chất phổ).

2. **Từ mô hình thính giác của tai người:**
   - Tai người cảm nhận tần số theo **thang phi tuyến (thang Mel)** — nghe rõ sự khác biệt ở tần số thấp hơn ở tần số cao.
   - Điều này dẫn đến việc chọn **MFCC** — đặc trưng mô phỏng cơ chế nghe của tai người trên thang Mel.

3. **Từ nghiên cứu khoa học (bài báo):**
   - MFCC được đề xuất bởi Davis & Mermelstein (1980) và đã trở thành **tiêu chuẩn vàng (gold standard)** trong nhận dạng giọng nói suốt 40 năm qua.
   - ZCR, Energy, Spectral Centroid là các đặc trưng cơ bản được sử dụng rộng rãi trong phân loại âm thanh (MPEG-7 Audio Descriptors).

4. **Từ thực nghiệm:**
   - Chạy thử với tập dữ liệu → đánh giá xem đặc trưng nào giúp phân biệt tốt (ví dụ: Centroid phân biệt giọng nam/nữ hiệu quả hơn ZCR).
   - Chuẩn hóa Z-Score rồi so sánh → nếu kết quả Top-3 hợp lý (cùng giới tính, cùng phong cách) thì đặc trưng đó phù hợp.

---

### Câu 14: Cơ sở dữ liệu quan hệ là gì?

**Trả lời:**

**Cơ sở dữ liệu quan hệ (Relational Database)** là mô hình cơ sở dữ liệu do **Edgar F. Codd** đề xuất năm 1970, trong đó dữ liệu được tổ chức thành các **bảng (relations/tables)**, và các bảng được liên kết với nhau thông qua **quan hệ logic (relationships)** dựa trên giá trị dữ liệu chung.

**Các thành phần cơ bản:**

| Thuật ngữ | Giải thích | Ví dụ trong project |
|---|---|---|
| **Relation (Bảng)** | Tập hợp các bản ghi cùng cấu trúc | `audio_metadata`, `audio_vectors` |
| **Tuple (Bộ/Hàng)** | Một bản ghi trong bảng | 1 dòng đại diện cho 1 file audio |
| **Attribute (Thuộc tính/Cột)** | Một trường dữ liệu | `filename`, `voice_vector`, `transcript` |
| **Primary Key (Khóa chính)** | Thuộc tính (hoặc nhóm thuộc tính) xác định duy nhất 1 bản ghi | `id` trong cả 2 bảng |
| **Foreign Key (Khóa ngoại)** | Thuộc tính tham chiếu đến khóa chính của bảng khác, tạo liên kết | `audio_vectors.id` → `audio_metadata.id` |
| **Domain (Miền giá trị)** | Tập hợp giá trị hợp lệ của thuộc tính | `id` ∈ {1, 2, ..., 510}, `filename` ∈ chuỗi ký tự |

**Đặc điểm chính:**
1. **Dữ liệu dạng bảng:** Mỗi bảng là một tập hợp các hàng (bản ghi) và cột (thuộc tính).
2. **Tính toàn vẹn:** Ràng buộc khóa chính đảm bảo không trùng lặp, khóa ngoại đảm bảo tham chiếu hợp lệ.
3. **Ngôn ngữ truy vấn SQL:** Sử dụng SQL (Structured Query Language) để thao tác dữ liệu (SELECT, INSERT, UPDATE, DELETE, JOIN...).
4. **Tính chất ACID:** Atomicity (nguyên tử), Consistency (nhất quán), Isolation (cô lập), Durability (bền vững) — đảm bảo dữ liệu luôn chính xác ngay cả khi có lỗi hệ thống.

**Áp dụng trong project:**
- Project sử dụng **PostgreSQL** — hệ quản trị CSDL quan hệ mã nguồn mở phổ biến nhất.
- 2 bảng `audio_metadata` và `audio_vectors` có quan hệ **1-1** qua khóa chính/khóa ngoại.
- Truy vấn tìm kiếm dùng SQL kết hợp **JOIN** 2 bảng và toán tử vector đặc biệt (`<->`, `<=>`) từ extension pgvector.

---

### Câu 15: Sau khi bọn em đọc các file vào thì trích xuất dữ liệu gì không?

**Trả lời:**

**Có.** Sau khi đọc các file `.wav` vào, hệ thống **trích xuất 2 loại dữ liệu đặc trưng:**

**Loại 1 — Đặc trưng Giọng nói (Voice Features) — 32 chiều:**

Từ tín hiệu âm thanh thô (raw signal), hệ thống trích xuất bằng xử lý tín hiệu số:
- **13 hệ số MFCC × 2 (Mean + Std) = 26 giá trị:** Mô tả hình bao phổ (spectral envelope) — bản "dấu vân tay âm thanh" của từng người nói.
- **ZCR × 2 (Mean + Std) = 2 giá trị:** Tỷ lệ đổi dấu tín hiệu — phân biệt âm hữu thanh/vô thanh.
- **RMS Energy × 2 (Mean + Std) = 2 giá trị:** Năng lượng trung bình — đo độ to/nhỏ giọng.
- **Spectral Centroid × 2 (Mean + Std) = 2 giá trị:** Tần số trọng tâm — giọng trầm/bổng.

→ Tổng cộng: **32 giá trị số thực** cho mỗi file, lưu vào `audio_voice_features.csv`.

**Loại 2 — Đặc trưng Nội dung (Content Features) — 384 chiều:**

Từ file audio, hệ thống dùng 2 model AI:
1. **Whisper (Speech-to-Text):** Chuyển âm thanh → văn bản (transcript).
2. **all-MiniLM-L6-v2 (Text Embedding):** Chuyển văn bản → vector 384 chiều trong không gian ngữ nghĩa.

→ Tổng cộng: **384 giá trị số thực** cho mỗi file, lưu vào `audio_content_features.csv`.

**Ngoài ra, hệ thống cũng trích xuất:**
- **Transcript (bản phiên âm):** Toàn bộ nội dung bài nói dạng văn bản tiếng Anh, lưu trong `metadata.csv` cột `transcript`.

**Tóm lại:** Mỗi file audio sau khi đọc vào sẽ sinh ra:
- 1 vector 32 chiều (đặc trưng vật lý)
- 1 vector 384 chiều (đặc trưng ngữ nghĩa)  
- 1 đoạn text transcript (nội dung bài nói)

Tất cả đều được lưu vào CSDL PostgreSQL để phục vụ tìm kiếm.

---

### Câu 16: Quy trình tìm kiếm được diễn ra như thế nào?

**Trả lời:**

Quy trình tìm kiếm diễn ra **6 bước tuần tự** như sau:

**Bước 1 — Người dùng upload file query:**
- Người dùng mở giao diện Streamlit (`app.py`), tải lên 1 file `.wav` bất kỳ.
- File được lưu vào thư mục tạm (temp) để xử lý.

**Bước 2 — Trích xuất đặc trưng giọng nói (Voice):**
- Gọi `extract_features(audio_path)` trong `feature_extractor.py`:
  - Đọc file WAV → Pre-emphasis → Framing (25ms/10ms) → Hamming Window → FFT (512 điểm) → Mel Filterbank (40 bộ lọc) → Log → DCT → 13 MFCC.
  - Tính thêm ZCR, RMS Energy, Spectral Centroid.
  - Lấy Mean + Std → **32 giá trị** (vector thô).
- **Chuẩn hóa Z-Score** bằng tham số μ, σ từ `scaler_params.json` → **Vector query 32 chiều đã chuẩn hóa**.

**Bước 3 — Trích xuất đặc trưng nội dung (Content):**
- Gọi `whisper_model.transcribe(audio_path)` → chuyển âm thanh thành **văn bản** (transcript).
- Gọi `st_model.encode(transcript)` → chuyển văn bản thành **vector 384 chiều** (text embedding).

**Bước 4 — Truy vấn CSDL tìm giọng nói giống nhất:**
- Gọi `search_voice(voice_vector, top_k=3)`:
  - Gửi câu SQL tới PostgreSQL, dùng toán tử `<->` (L2 Distance) để tính khoảng cách Euclidean giữa query vector và tất cả `voice_vector` trong bảng `audio_vectors`.
  - Chỉ mục **HNSW** giúp tìm nhanh mà không cần duyệt tuần tự.
  - Tính `Score = 1 / (1 + distance/10)` → sắp xếp giảm dần → lấy **Top 3**.

**Bước 5 — Truy vấn CSDL tìm nội dung giống nhất:**
- Gọi `search_content(content_vector, top_k=3)`:
  - Gửi câu SQL, dùng toán tử `<=>` (Cosine Distance) giữa query embedding và tất cả `content_vector`.
  - Tính `Score = 1 - Cosine Distance` → sắp xếp giảm dần → lấy **Top 3**.

**Bước 6 — Hiển thị kết quả:**
- Giao diện Streamlit hiển thị **2 nhóm kết quả** song song:
  - **Top 3 Voice:** Tên file + % Voice Similarity + Audio Player (nghe lại).
  - **Top 3 Content:** Tên file + % Content Similarity + Trích đoạn transcript + Nút xem đầy đủ.

**Toàn bộ quy trình mất khoảng 5-15 giây** (phần lớn thời gian dành cho Whisper transcribe). Phần truy vấn CSDL chỉ mất vài mili-giây nhờ chỉ mục HNSW.

---

### Câu 17: Tần số lấy mẫu của file âm thanh là bao nhiêu?

**Trả lời:**

**Tần số lấy mẫu (Sample Rate) là 16,000 Hz (16 kHz).**

**Giải thích:**
- Tần số lấy mẫu 16,000 Hz nghĩa là mỗi giây, tín hiệu âm thanh được đo **16,000 lần** (16,000 mẫu/giây).
- Mỗi mẫu là một giá trị số nguyên 16-bit (`int16`, phạm vi -32768 đến 32767) biểu diễn biên độ sóng âm tại thời điểm lấy mẫu đó.

**Tại sao là 16 kHz:**
- Theo **Định lý Nyquist-Shannon**, tần số lấy mẫu phải ≥ 2× tần số cao nhất cần thu. Giọng nói con người có dải tần hữu ích nằm trong khoảng **100 Hz - 8,000 Hz**.
- Với sample rate 16 kHz, ta thu được tín hiệu đến tối đa **8,000 Hz** (= 16000/2, gọi là tần số Nyquist), vừa đủ để bao phủ toàn bộ dải tần giọng nói.
- 16 kHz là chuẩn **"wideband speech"** — đủ chất lượng cho nhận dạng giọng nói (ASR) mà không lãng phí dung lượng. Đây cũng là tần số đầu vào mặc định của model **Whisper** (OpenAI).

**Ví dụ cụ thể:**
- File `GapPidut5nk.wav`: Sample Rate = 16,000 Hz, có 1,654,443 mẫu → thời lượng = 1,654,443 / 16,000 = **103.40 giây** (~1 phút 43 giây).
- Kích thước file ≈ 1,654,443 × 2 byte (16-bit) + 44 byte header = ~3.2 MB.

**Trong code, tần số lấy mẫu được đọc tự động:**
```python
# feature_extractor.py, dòng 50
sample_rate, signal = wavfile.read(filepath)
# sample_rate = 16000
```
Giá trị `sample_rate` sau đó được dùng để:
- Tính kích thước frame: `frame_length = 0.025 * 16000 = 400 mẫu` (25ms).
- Tính bước nhảy: `frame_step = 0.010 * 16000 = 160 mẫu` (10ms).
- Tạo bộ lọc Mel: `get_filterbanks(nfilt=40, nfft=512, samplerate=16000)`.
- Tính Spectral Centroid: `freqs = np.fft.rfftfreq(512, d=1.0/16000)`.

---

## 11. Cài đặt & Chạy lại từ Đầu

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
