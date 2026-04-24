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
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_eda_basic(df, text_col, label_col):
    print("\nMelakukan Exploratory Data Analysis (EDA)...")
    
    print("Missing Values:")
    print(df.isnull().sum())
    
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=label_col, order=df[label_col].value_counts().index, hue=label_col, legend=False, palette='viridis')
    plt.title('Distribusi Kelas Emosi')
    plt.xlabel('Emosi')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/images/distribusi_emosi.png')
    
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True, color='blue')
    plt.title('Distribusi Panjang Teks')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    plt.tight_layout()
    plt.savefig('results/images/distribusi_panjang_teks.png')
    
    return df

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower()
    
    # Remove mentions, URLs, HTML tags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Expand contractions
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)
    
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Reduce repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'im', 'dont', 'cant', 'didnt', 'doesnt', 'just', 'like', 'get', 'go', 'know', 'amp', 'got'}
    stop_words = stop_words.union(custom_stopwords)
    
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_data(df, text_col):
    print("\nMemulai preprocessing teks...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    print("Preprocessing selesai.")
    return df

def plot_top_words_per_emotion(df, text_col, label_col):
    print("\nMengekstrak kata kunci per emosi (TF-IDF)...")
    emotions = df[label_col].unique()
    
    docs = []
    emotion_list = []
    for emotion in emotions:
        text_data = df[df[label_col] == emotion][text_col].dropna().astype(str)
        docs.append(" ".join(text_data))
        emotion_list.append(emotion)
        
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    
    n_emotions = len(emotions)
    cols = 3
    rows = (n_emotions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()
    
    for i, emotion in enumerate(emotion_list):
        row_scores = tfidf_matrix[i].toarray()[0]
        top_indices = row_scores.argsort()[-10:][::-1]
        
        words = [feature_names[j] for j in top_indices]
        scores = [row_scores[j] for j in top_indices]
        
        if sum(scores) > 0:
            axes[i].barh(words, scores, color='teal')
            axes[i].invert_yaxis()
            axes[i].set_title(f'Top Words: {emotion.upper()}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Skor TF-IDF')
        else:
            axes[i].set_title(f'Top Words: {emotion.upper()} (Kosong)')
            
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig('results/images/top_words_per_emotion.png')
    
    print("\nMembuat WordCloud dari data keseluruhan...")
    text_all = " ".join(df[text_col].dropna().astype(str))
    if text_all.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_all)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud: Data Teks Bersih', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/images/wordcloud_all_cleaned.png')

if __name__ == "__main__":
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    
    if os.path.exists('dataset/combined_emotions.csv'):
        file_name = 'dataset/combined_emotions.csv'
    else:
        file_name = 'dataset/tweet_emotions.csv'
        
    text_col_name = 'content'
    label_col_name = 'sentiment'
    
    df = load_data(file_name)
    
    if df is not None:
        df = perform_eda_basic(df, text_col_name, label_col_name)
        df_clean = preprocess_data(df, text_col_name)
        
        plot_top_words_per_emotion(df_clean, 'cleaned_text', label_col_name)
        
        output_file = 'results/data/dataset_cleaned.csv'
        df_clean.to_csv(output_file, index=False)

# ==========================================
# AIRFLOW DAG DEFINITION
# ==========================================
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta

    default_args = {
        'owner': 'data_team',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        'eda_preprocessing_pipeline',
        default_args=default_args,
        description='DAG untuk otomatisasi EDA dan Preprocessing',
        schedule_interval=timedelta(days=1),
        catchup=False
    )

    def task_eda_basic():
        os.makedirs('results/images', exist_ok=True)
        os.makedirs('results/data', exist_ok=True)
        file_name = 'dataset/combined_emotions.csv' if os.path.exists('dataset/combined_emotions.csv') else 'dataset/tweet_emotions.csv'
        df = load_data(file_name)
        if df is not None:
            perform_eda_basic(df, 'content', 'sentiment')

    def task_preprocessing():
        os.makedirs('results/data', exist_ok=True)
        file_name = 'dataset/combined_emotions.csv' if os.path.exists('dataset/combined_emotions.csv') else 'dataset/tweet_emotions.csv'
        df = load_data(file_name)
        if df is not None:
            df_clean = preprocess_data(df, 'content')
            df_clean.to_csv('results/data/dataset_cleaned.csv', index=False)

    def task_eda_advanced():
        cleaned_file = 'results/data/dataset_cleaned.csv'
        if os.path.exists(cleaned_file):
            df_clean = pd.read_csv(cleaned_file)
            plot_top_words_per_emotion(df_clean, 'cleaned_text', 'sentiment')
        else:
            print("File data bersih tidak ditemukan.")

    t1 = PythonOperator(
        task_id='run_eda_basic',
        python_callable=task_eda_basic,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id='run_text_preprocessing',
        python_callable=task_preprocessing,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id='run_eda_advanced',
        python_callable=task_eda_advanced,
        dag=dag,
    )

    # Workflow dependencies
    t1 >> t2 >> t3

except ImportError:
    # Mengabaikan error jika Apache Airflow tidak terinstall di environment saat ini.
    pass
