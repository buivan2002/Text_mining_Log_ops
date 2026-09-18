"""Mô hình baseline mốc: TF-IDF + SVM để đối chiếu hiệu năng với pipeline chính."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


class BaselineTfidfSvm:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
