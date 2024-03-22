import numpy as np
import pandas as pd
import pickle
import math
from collections import Counter

class TextVectorizer:
    def __init__(self):
        self.vocab = None
        self.idf_dict = None

    def fit_transform(self, corpus):
        self.vocab = self.create_vocabulary(corpus)
        self.idf_dict = self.calculate_idf(list(self.vocab.keys()), corpus)
        tfidf_matrix = self.convert_to_tfidf(corpus, self.vocab, self.idf_dict)
        return tfidf_matrix

    def create_vocabulary(self, preprocessed_texts):
        word_set = set()
        if isinstance(preprocessed_texts, list):
            for text in preprocessed_texts:
                if text is not None:
                    for word in text.split():
                        word_set.add(word)
            sorted_words = sorted(list(word_set))
            vocabulary = {word: idx for idx, word in enumerate(sorted_words)}
            return vocabulary
        else:
            print("Incorrect format.")

    def calculate_idf(self, unique_words, preprocessed_texts):
        idf_values = {}
        num_docs = len(preprocessed_texts)
        for word in unique_words:
            word_count = sum(1 for doc in preprocessed_texts if doc and word in doc)
            idf_values[word] = float(1 + math.log((num_docs + 1) / (word_count + 1)))
        return idf_values

    def convert_to_tfidf(self, preprocessed_texts, vocab, idf_values):
        if isinstance(preprocessed_texts, list):
            tfidf_matrix = np.zeros((len(preprocessed_texts), len(vocab)))
            for idx, text in enumerate(preprocessed_texts):
                if text is not None:
                    word_freq = dict(Counter(text.split()))
                    for word, freq in word_freq.items():
                        col_index = vocab.get(word, -1)
                        if col_index != -1:
                            tf = freq / float(len(text.split()))
                            idf_ = idf_values[word]
                            tfidf_matrix[idx, col_index] = tf * idf_
            norms = np.linalg.norm(tfidf_matrix, axis=1)[:, np.newaxis]
            zero_indices = np.where(norms == 0)[0]
            norms[zero_indices] = 1
            norms[np.isnan(norms)] = 0
            tfidf_matrix /= norms
            return tfidf_matrix
        else:
            print("Incorrect format.")

# Load the CSV file
df = pd.read_csv('preprocessed_text_data.csv')
df = df.dropna(subset=['Preprocessed Review'])

# Define the corpus as the whole column 'Preprocessed Review'
corpus = df['Preprocessed Review'].tolist()

# Instantiate TextVectorizer
text_vectorizer = TextVectorizer()

# Calculating the TF-IDF matrix
tfidf_matrix = text_vectorizer.fit_transform(corpus)

# Converting tfidf values to comma separated
tfidf_vectors = [','.join(map(str, row)) for row in tfidf_matrix]

# Add the merged TF-IDF vectors as a new column in the DataFrame
df['TF-IDF'] = tfidf_vectors

# Saving the tfidf vector
df.to_csv("tfidf_score.csv", index=False)
print("TF-IDF scores saved to tfidf_score.csv")

# Saving the tfidf values in a pickle file
df.to_pickle("tfidf_score.pkl")
print("TF-IDF scores saved to tfidf_score.pkl")
