import pandas as pd
import os
import glob

def merge_datasets():
    print("Mencari dan menggabungkan dataset...")
    
    txt_files = glob.glob("dataset/*.txt")
    txt_files = [f for f in txt_files if f != 'requirements.txt']
    
    if not txt_files:
        print("Dataset teks tidak ditemukan.")
        return

    dfs = []
    
    old_file = 'dataset/tweet_emotions.csv'
    if os.path.exists(old_file):
        print("Memuat dataset awal...")
        df_old = pd.read_csv(old_file)
        if 'sentiment' in df_old.columns and 'content' in df_old.columns:
            df_old = df_old[['sentiment', 'content']]
            dfs.append(df_old)

    print("Memproses file dataset teks...")
    for file in txt_files:
        try:
            df_new = pd.read_csv(file, sep=';', names=['content', 'sentiment'])
            df_new = df_new[['sentiment', 'content']]
            dfs.append(df_new)
        except Exception as e:
            print(f"Gagal membaca {file}: {e}")

    if dfs:
        print("Menggabungkan data...")
        df_combined = pd.concat(dfs, ignore_index=True)
        
        total_sebelum = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        total_sesudah = len(df_combined)
        
        if total_sebelum - total_sesudah > 0:
            print(f"Dihapus {total_sebelum - total_sesudah} baris duplikat.")
            
        output_file = 'dataset/combined_emotions.csv'
        df_combined.to_csv(output_file, index=False)
        print(f"Dataset berhasil disimpan ke '{output_file}' dengan total {len(df_combined)} baris.")
    else:
        print("Tidak ada data yang diproses.")

if __name__ == "__main__":
    merge_datasets()
