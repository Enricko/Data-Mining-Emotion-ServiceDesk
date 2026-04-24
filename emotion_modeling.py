import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_extract_emotions(file_path, text_col='cleaned_text', label_col='sentiment'):
    print(f"Memuat data dari {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Dataset tidak ditemukan.")
        return

    df = df.dropna(subset=[text_col, label_col])

    print("\nMengekstraksi fitur teks menggunakan TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nMelatih model Logistic Regression (dengan penyeimbangan kelas otomatis)...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, multi_class='ovr')
    model.fit(X_train, y_train)

    print("\nHasil Evaluasi Model Pada Data Test:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nKata Kunci Paling Berpengaruh Untuk Setiap Emosi:")
    feature_names = vectorizer.get_feature_names_out()
    
    for i, class_label in enumerate(model.classes_):
        coefs = model.coef_[i]
        top_indices = coefs.argsort()[-15:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        
        print(f"[{class_label.upper()}]: {', '.join(top_words)}")

if __name__ == "__main__":
    train_and_extract_emotions('results/data/dataset_cleaned.csv')
