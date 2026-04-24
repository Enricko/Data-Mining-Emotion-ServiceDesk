from datasets import load_dataset
import pandas as pd
import os

def download_and_save():
    print("Mendownload dataset 'Emotions dataset for NLP' (Praveen Govi / dair-ai) dari HuggingFace...")
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    
    # Mapping label HuggingFace (angka) ke label teks yang sesuai dengan dataset Anda
    label_mapping = {
        0: 'sadness',
        1: 'happiness', # 'joy' diubah menjadi 'happiness' agar gabung dengan dataset lama
        2: 'love',
        3: 'anger',
        4: 'worry',     # 'fear' diubah menjadi 'worry'
        5: 'surprise'
    }
    
    for split in ['train', 'validation', 'test']:
        print(f"Menyimpan {split}.txt...")
        df = pd.DataFrame(dataset[split])
        # Map angka ke nama emosi
        df['label_text'] = df['label'].map(label_mapping)
        
        # Format Praveen Govi: text;emotion
        filename = f"{split if split != 'validation' else 'val'}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for text, emotion in zip(df['text'], df['label_text']):
                f.write(f"{text};{emotion}\n")
                
    print("Download selesai! File train.txt, val.txt, test.txt berhasil dibuat.")
    
if __name__ == "__main__":
    download_and_save()
