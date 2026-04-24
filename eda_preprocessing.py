import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Fix for punkt downloading in some environments
nltk.download('punkt_tab') 

def load_data(file_path):
    """
    Memuat dataset dari file CSV.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_eda(df, text_col, label_col):
    """
    Melakukan Exploratory Data Analysis (EDA).
    """
    print("\n--- Memulai Exploratory Data Analysis (EDA) ---")
    
    # 1. Cek missing values
    print("Missing Values per kolom:")
    print(df.isnull().sum())
    
    # 2. Hapus baris yang memiliki missing value pada teks atau label (opsional, tapi disarankan)
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # 3. Distribusi Kelas Emosi
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=label_col, order=df[label_col].value_counts().index, palette='viridis')
    plt.title('Distribusi Kelas Emosi')
    plt.xlabel('Emosi')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('distribusi_emosi.png')
    print("Grafik disimpan sebagai 'distribusi_emosi.png'")
    
    # 4. Distribusi Panjang Teks (Jumlah Kata)
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True, color='blue')
    plt.title('Distribusi Panjang Teks (Jumlah Kata)')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    plt.tight_layout()
    plt.savefig('distribusi_panjang_teks.png')
    print("Grafik disimpan sebagai 'distribusi_panjang_teks.png'")
    
    # 5. WordCloud untuk teks secara keseluruhan
    text_all = " ".join(review for review in df[text_col].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_all)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud: Kata-kata Paling Sering Muncul')
    plt.tight_layout()
    plt.savefig('wordcloud_all.png')
    print("WordCloud disimpan sebagai 'wordcloud_all.png'")

    return df

def clean_text(text):
    """
    Fungsi preprocessing teks.
    """
    if not isinstance(text, str):
        text = str(text)
        
    # a. Lowercasing
    text = text.lower()
    
    # b. Menghapus URL, tag HTML, angka dan karakter khusus
    text = re.sub(r'http\S+|www\.\S+', '', text) # Hapus URL
    text = re.sub(r'<.*?>', '', text) # Hapus HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus angka dan punctuation
    
    # c. Tokenisasi
    tokens = word_tokenize(text)
    
    # d. Hapus Stopwords (menggunakan english, ganti 'indonesian' jika dataset berbahasa Indonesia)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # e. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_data(df, text_col):
    """
    Menerapkan transformasi preprocessing text.
    """
    print("\n--- Memulai Preprocessing Data ---")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    print("Preprocessing selesai. Kolom 'cleaned_text' telah dibuat.")
    return df

if __name__ == "__main__":
    # Menentukan nama file dan kolom sesuai dataset "Emotion Detection from Text" (oleh pashupatigupta)
    file_name = 'tweet_emotions.csv'
    text_col_name = 'content'
    label_col_name = 'sentiment'
    
    # Memeriksa apakah dataset sudah ada di folder ini:
    if not os.path.exists(file_name):
        print(f"File {file_name} tidak ditemukan di folder ini!")
        print("Silakan download dataset dari: https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text")
        print(f"Ekstrak dan simpan file '{file_name}' di folder yang sama dengan script ini.")
        print("\nUntuk sementara waktu agar script bisa berjalan, saya membuatkan 5 baris data dummy.\n")
        
        dummy_data = {
            'tweet_id': [1956967341, 1956967666, 1956967696, 1956967789, 1956968416],
            'sentiment': ['empty', 'sadness', 'sadness', 'enthusiasm', 'neutral'],
            'content': [
                "@tiffanylue i know  i was listenin to bad habit earlier and i started freakin out.",
                "Layin n bed with a headache  ughhhh...waitin on your call...",
                "Funeral ceremony...gloomy friday...",
                "wants to hang out with friends SOON!",
                "@dannycastillo We want to trade with someone who has Houston tickets, but no one will."
            ]
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy.to_csv(file_name, index=False)

    # 1. Proses Muat Data
    df = load_data(file_name)
    
    if df is not None:
        # 2. EDA (Exploratory Data Analysis)
        df = perform_eda(df, text_col_name, label_col_name)
        
        # 3. Preprocessing Data Teks
        df_clean = preprocess_data(df, text_col_name)
        
        print("\nContoh Hasil (Sebelum vs Sesudah Preprocessing):")
        print(df_clean[[text_col_name, 'cleaned_text']].head())
        
        # 4. Simpan Data Bersih Ke File Baru
        output_file = 'dataset_cleaned.csv'
        df_clean.to_csv(output_file, index=False)
        print(f"\nBerhasil! Dataset yang sudah di-preprocess telah disimpan ke '{output_file}'")
