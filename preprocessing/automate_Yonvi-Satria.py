# automate_Nama-siswa.py
# Penulis: [Nama Anda]
# Deskripsi: Script untuk preprocessing otomatis dataset concrete.csv

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_concrete_dataset(
    input_path="concrete/concrete.csv",
    output_dir="preprocessing/concrete_preprocessed"
):
    # Cek apakah file tersedia
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {input_path}")
    
    # Load dataset
    df = pd.read_csv(input_path)

    # Cek kolom target
    target_col = "strength"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

    # Pisahkan fitur dan target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Simpan hasil ke CSV
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"✅ Preprocessing selesai. Hasil disimpan di folder '{output_dir}'.")

# Jika dijalankan langsung
if __name__ == "__main__":
    preprocess_concrete_dataset()
