import pandas as pd
import docx
import os

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f" Помилка DOCX {path}: {e}")
        return None

def extract_text_from_txt(path):
    for enc in ['utf-8', 'cp1251', 'latin-1']:
        try:
            with open(path, "r", encoding=enc, errors='ignore') as f:
                return f.read()
        except:
            continue
    return None

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(current_dir, "dataset_text_links.csv")
    output_csv = os.path.join(current_dir, "dataset_text.csv")

    if not os.path.exists(input_csv):
        print(f" не знайдено: {input_csv}")
        return

    
    lines = []
    try:
        with open(input_csv, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except:
        with open(input_csv, "r", encoding="cp1251", errors="ignore") as f:
            lines = f.readlines()

    dataset = []

    for line in lines:
        line = line.strip()
        if not line or "path,label" in line:
            continue
            
        if "," in line:
            parts = line.rsplit(",", 1)
            raw_path = parts[0].strip()
            label = parts[1].strip()
        
            file_path = os.path.join(current_dir, raw_path.replace("/", os.sep))

            if os.path.exists(file_path):
                text = None
                if file_path.lower().endswith(".txt"):
                    text = extract_text_from_txt(file_path)
                elif file_path.lower().endswith(".docx"):
                    text = extract_text_from_docx(file_path)

                if text:
                    dataset.append({"text": text.strip(), "label": label})
            else:
                print(f"Файл не знайдено: {raw_path}")

    if dataset:
        df_result = pd.DataFrame(dataset)
        # Зберігаємо в UTF-8-SIG (найкраще для Windows/Excel)
        df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print("-" * 30)
        print(f"створено: {output_csv}")
        print(f"текстів: {len(dataset)}")
    else:
        print("не вдалося")

if __name__ == "__main__":
    main()

