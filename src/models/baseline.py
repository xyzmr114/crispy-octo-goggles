"""
Traditional NLP Baselines:
TF-IDF + Logistic Regression & Linear Support Vector Machine (LinearSVC)
Serves as the empirical benchmark to contextualize deep neural gains.
"""

import json
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix


def load_texts_and_labels(jsonl_path: str, mode: str = "output") -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if mode == "output":
                    texts.append(item["llm_output"])
                elif mode == "input":
                    texts.append(item["llm_input"])
                else:
                    texts.append(item["llm_input"] + " " + item["llm_output"])
                labels.append(item["family"])
    return texts, labels


class BaselineSuite:
    def __init__(self, max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode"
        )
        self.lr_model = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        self.svm_model = LinearSVC(max_iter=2000, C=1.0, random_state=42)
        self.is_fitted = False

    def fit(self, train_texts: List[str], train_labels: List[str]):
        print(f"Fitting TF-IDF Vectorizer on {len(train_texts)} texts...")
        X_train = self.vectorizer.fit_transform(train_texts)
        
        print("Training Logistic Regression...")
        self.lr_model.fit(X_train, train_labels)
        
        print("Training Linear SVM...")
        self.svm_model.fit(X_train, train_labels)
        self.is_fitted = True

    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Dict]:
        assert self.is_fitted, "Models must be fitted before evaluation"
        X_test = self.vectorizer.transform(test_texts)
        
        # Logistic Regression
        lr_preds = self.lr_model.predict(X_test)
        lr_acc = accuracy_score(test_labels, lr_preds)
        lr_f1 = f1_score(test_labels, lr_preds, average="macro")
        lr_report = classification_report(test_labels, lr_preds, output_dict=True)
        lr_cm = confusion_matrix(test_labels, lr_preds)
        
        # Linear SVM
        svm_preds = self.svm_model.predict(X_test)
        svm_acc = accuracy_score(test_labels, svm_preds)
        svm_f1 = f1_score(test_labels, svm_preds, average="macro")
        svm_report = classification_report(test_labels, svm_preds, output_dict=True)
        svm_cm = confusion_matrix(test_labels, svm_preds)
        
        return {
            "logistic_regression": {
                "accuracy": lr_acc,
                "macro_f1": lr_f1,
                "report": lr_report,
                "confusion_matrix": lr_cm.tolist(),
                "predictions": lr_preds.tolist()
            },
            "linear_svm": {
                "accuracy": svm_acc,
                "macro_f1": svm_f1,
                "report": svm_report,
                "confusion_matrix": svm_cm.tolist(),
                "predictions": svm_preds.tolist()
            }
        }
