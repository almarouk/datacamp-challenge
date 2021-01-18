from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pandas

class CustomClassifier(BaseEstimator):
    """Dummy detector which ouputs 1-second long steps every two seconds."""

    def __init__(self):
    	self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    	self.clf = LogisticRegression(random_state=42)

    def fit(self, X, y):
        embeddings = self.model.encode(list(X['Text']), show_progress_bar=True)
        self.clf.fit(embeddings, y)
        return self

    def predict(self, X):
        return self.clf.predict(self.model.encode(list(X['Text'])))


def get_estimator():

    return CustomClassifier()
