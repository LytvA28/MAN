import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не знайдено: {path}")

    df = pd.read_csv(path)
    
    df = df.dropna(subset=["text", "label"])

def build_vectorizer():
    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode"
    )


def build_model():
    return LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        n_jobs=-1
    )


def train_and_evaluate(X, y, vectorizer, model):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report


def save_artifacts(model, vectorizer, folder):
    
    model_path = os.path.join(folder, "text_model.pkl")
    vectorizer_path = os.path.join(folder, "text_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"збережено: {model_path}")
    print(f"збережено: {vectorizer_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset_text.csv")

    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        print(f"error: {e}")
        return

    print(f"текстів: {len(df)}")

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = build_model()

    accuracy, report = train_and_evaluate(X, y, vectorizer, model)


    save_artifacts(model, vectorizer, base_dir)


if __name__ == "__main__":
    main()

