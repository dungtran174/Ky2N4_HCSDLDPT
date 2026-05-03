import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import pandas as pd
import os
import matplotlib.pyplot as plt

def hz_to_mel(hz):
    """Chuyển đổi tần số Hz sang thang đo Mel"""
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    """Chuyển đổi thang đo Mel về tần số Hz"""
    return 700 * (10**(mel / 2595.0) - 1)

def get_filterbanks(nfilt=40, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Tạo bộ lọc Mel (Mel Filterbank)
    Gồm nfilt bộ lọc tam giác xếp chồng lên nhau trên thang đo Mel.
    """
    highfreq = highfreq or samplerate / 2
    low_mel = hz_to_mel(lowfreq)
    high_mel = hz_to_mel(highfreq)
    
    # Chia đều khoảng cách trên thang Mel
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Chuyển đổi tần số Hz sang chỉ số bin của FFT
    bin = np.floor((nfft + 1) * hz_points / samplerate)
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # Điểm bắt đầu của bộ lọc (trái)
        f_m = int(bin[m])             # Đỉnh của bộ lọc (giữa)
        f_m_plus = int(bin[m + 1])    # Điểm kết thúc của bộ lọc (phải)
        
        # Sườn dốc lên
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        # Sườn dốc xuống
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    return fbank

def extract_features(filepath):
    # 1. Đọc file
    # Sử dụng scipy.io.wavfile.read
    sample_rate, signal = wavfile.read(filepath)
    
    # Nếu là stereo (2 kênh), chuyển về mono bằng trung bình cộng
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)
        
    signal = signal.astype(float)
    
    # 2. Tiền xử lý: Pre-emphasis
    # Công thức: y[n] = x[n] - 0.97 * x[n-1]
    # Tác dụng: Khuếch đại tần số cao, cân bằng phổ do giọng nói tự nhiên bị giảm năng lượng ở tần số cao.
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # 3. Phân đoạn (Framing)
    # Chia tín hiệu thành các khung 25ms, độ dịch 10ms
    frame_size = 0.025 # 25ms
    frame_stride = 0.010 # 10ms
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    # Zero padding để đảm bảo tất cả frame có cùng độ dài
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    
    # Lấy chỉ số các mẫu cho từng khung
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # 4. Áp dụng cửa sổ Hamming
    # Công thức: w[n] = 0.54 - 0.46 * cos(2*pi*n / (M-1))
    # Tác dụng: Giảm hiện tượng rò rỉ phổ (spectral leakage) ở hai mép của mỗi khung.
    M = frame_length
    hamming_window = 0.54 - 0.46 * np.cos((2 * np.pi * np.arange(M)) / (M - 1))
    frames *= hamming_window
    
    # 5. Trích xuất MFCC
    # 5.1. FFT (512 điểm) -> Power Spectrum (Phổ công suất)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))    # Power Spectrum
    
    # 5.2. Áp dụng Mel Filterbank (40 bộ lọc tam giác)
    nfilt = 40
    fbank = get_filterbanks(nfilt, NFFT, sample_rate)
    filter_banks = np.dot(pow_frames, fbank.T)
    # Tránh lỗi log(0) bằng cách cộng eps
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) 
    
    # 5.3. Logarithm năng lượng dải Mel (Mô phỏng cảm nhận độ to âm thanh của tai người)
    filter_banks = 10 * np.log10(filter_banks)
    
    # 5.4. Thực hiện DCT (Discrete Cosine Transform) lấy 13 hệ số đầu tiên
    # Tại sao lại dùng DCT 13 hệ số? Vì các hệ số dải mel thường có tương quan với nhau cao,
    # DCT giúp khử tương quan và dồn năng lượng thông tin vào các hệ số đầu tiên.
    # 13 hệ số đầu đã đủ để mô tả hình bao phổ (spectral envelope) của giọng nói.
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    
    # 6. Các đặc trưng bổ trợ
    # 6.1. ZCR (Zero Crossing Rate): Tỉ lệ đổi dấu của tín hiệu trong khung
    # Phản ánh độ ồn (noisiness), hữu ích phân biệt giọng hữu thanh (vowel) và vô thanh (consonant).
    sgn_frames = np.sign(frames)
    zcr = np.mean(np.abs(sgn_frames[:, 1:] - sgn_frames[:, :-1]), axis=1) / 2.0
    
    # 6.2. RMS Energy: Căn bậc hai trung bình bình phương biên độ
    # Đo năng lượng (độ lớn) thực tế của khung tín hiệu.
    rms_energy = np.sqrt(np.mean(frames**2, axis=1))
    
    # 6.3. Spectral Centroid: Trọng tâm phổ
    # Thể hiện "độ sáng" của âm thanh.
    freqs = np.fft.rfftfreq(NFFT, d=1.0/sample_rate)
    # Cân bằng tránh chia 0
    spectral_centroid = np.sum(freqs * mag_frames, axis=1) / (np.sum(mag_frames, axis=1) + np.finfo(float).eps)
    
    # 7. Tính Mean và Std (Tổng 32 đặc trưng)
    features = {}
    for i in range(num_ceps):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[:, i])
        features[f'mfcc_{i+1}_std'] = np.std(mfcc[:, i])
        
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    features['energy_mean'] = np.mean(rms_energy)
    features['energy_std'] = np.std(rms_energy)
    
    features['centroid_mean'] = np.mean(spectral_centroid)
    features['centroid_std'] = np.std(spectral_centroid)
    
    return features

def plot_check(filepath):
    """
    Hàm trực quan hóa Waveform và Spectrogram để phục vụ giải trình "so sánh cảm tính".
    """
    sample_rate, signal = wavfile.read(filepath)
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)
        
    plt.figure(figsize=(12, 8))
    
    # Vẽ Waveform
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, color='b')
    plt.title(f'Waveform of {os.path.basename(filepath)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Vẽ Spectrogram
    plt.subplot(2, 1, 2)
    # Dùng hàm specgram mặc định của matplotlib với Hamming window
    plt.specgram(signal, Fs=sample_rate, NFFT=512, noverlap=256, cmap='viridis')
    plt.title(f'Spectrogram of {os.path.basename(filepath)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    
    plt.tight_layout()
    plt.show()

def process_all(metadata_path, output_path):
    # Đọc danh sách từ metadata.csv
    df = pd.read_csv(metadata_path)
    
    all_features = []
    
    for idx, row in df.iterrows():
        filepath = row['filepath']
        try:
            feats = extract_features(filepath)
            # Thêm khóa chính từ metadata
            feats['file_id'] = row['file_id']
            feats['filename'] = row['filename']
            all_features.append(feats)
            
            # Log tiến trình
            if (idx + 1) % 10 == 0:
                print(f"Đã xử lý {idx + 1}/{len(df)} files.")
        except Exception as e:
            print(f"Lỗi khi xử lý {filepath}: {e}")
            
    df_features = pd.DataFrame(all_features)
    
    # Đảm bảo thứ tự các cột đúng như yêu cầu
    cols = ['file_id', 'filename']
    for i in range(1, 14):
        cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    cols.extend(['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'centroid_mean', 'centroid_std'])
    
    df_features = df_features[cols]
    
    # Lưu kết quả
    df_features.to_csv(output_path, index=False)
    print(f"Hoàn thành! Đã lưu đặc trưng của {len(all_features)} files vào {output_path}")

if __name__ == '__main__':
    metadata_csv = 'metadata.csv'
    output_csv = 'audio_voice_features.csv'
    
    print("Bắt đầu quá trình trích xuất đặc trưng (KHÔNG dùng librosa)...")
    process_all(metadata_csv, output_csv)
    
    # Ví dụ gọi hàm test trực quan hóa (bỏ comment để dùng thử)
    # plot_check('data/xkE-CLCi-k8.wav')
