import pandas as pd
import json

def normalize_voice_features():
    print("1. Reading raw data...")
    df_voice = pd.read_csv('audio_voice_features.csv')
    
    # Liệt kê 32 cột đặc trưng
    voice_cols = []
    for i in range(1, 14):
        voice_cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    voice_cols.extend(['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'centroid_mean', 'centroid_std'])
    
    print("2. Calculating Mean and Std...")
    mean_vals = df_voice[voice_cols].mean().to_dict()
    std_vals = df_voice[voice_cols].std().to_dict()
    
    # Lưu tham số chuẩn hóa để Query dùng lại (RẤT QUAN TRỌNG)
    with open('scaler_params.json', 'w') as f:
        json.dump({'mean': mean_vals, 'std': std_vals}, f, indent=4)
        
    print("3. Scaling data...")
    for col in voice_cols:
        # Nếu std = 0 thì không chia để tránh lỗi
        if std_vals[col] != 0:
            df_voice[col] = (df_voice[col] - mean_vals[col]) / std_vals[col]
            
    # Lưu ra file mới
    df_voice.to_csv('audio_voice_features_scaled.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    normalize_voice_features()
