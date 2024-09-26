import os
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class VectorSpaceModel:
    def _init_(self):
        self.docs = {}
        self.index = defaultdict(dict)
        self.idf_values = {}
        self.doc_lengths = {}
        self.N = 0
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.read_documents()

    def read_documents(self):
        dir_path = './'
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_id = filename.replace('.txt', '')
                    self.docs[doc_id] = content
                    tokens = self.preprocess(content)
                    self.build_inverted_index(tokens, doc_id)
        self.N = len(self.docs)
        self.calculate_idf()
        self.calculate_doc_lengths()

    def preprocess(self, content):
        tokens = word_tokenize(content.lower())
        tokens = [token for token in tokens if token not in self.punctuation and token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def build_inverted_index(self, tokens, doc_id):
        for term in set(tokens):
            if doc_id not in self.index[term]:
                self.index[term][doc_id] = tokens.count(term)

    def calculate_idf(self):
        for term, doc_dict in self.index.items():
            self.idf_values[term] = math.log10(self.N / len(doc_dict))

    def calculate_doc_lengths(self):
        for doc_id in self.docs:
            length = 0
            for term, doc_dict in self.index.items():
                if doc_id in doc_dict:
                    tf = doc_dict[doc_id] / len(self.preprocess(self.docs[doc_id]))
                    length += (tf * self.idf_values[term]) ** 2
            self.doc_lengths[doc_id] = math.sqrt(length)

    def tf_idf(self, term, doc_id):
        if term in self.index and doc_id in self.index[term]:
            tf = self.index[term][doc_id] / len(self.preprocess(self.docs[doc_id]))
            return tf * self.idf_values[term]
        return 0

    def cosine_similarity(self, query_vector, doc_id):
        numerator = 0
        for term, query_weight in query_vector.items():
            numerator += query_weight * self.tf_idf(term, doc_id)
        
        doc_length = self.doc_lengths[doc_id]
        query_length = math.sqrt(sum(w**2 for w in query_vector.values()))
        
        if query_length == 0 or doc_length == 0:
            return 0
        return numerator / (query_length * doc_length)

    def query_vector(self, query):
        terms = self.preprocess(query)
        query_vector = {}
        for term in set(terms):
            tf = terms.count(term) / len(terms)
            if term in self.idf_values:
                query_vector[term] = tf * self.idf_values[term]
        return query_vector

    def rank_documents(self, query):
        query_vector = self.query_vector(query)
        scores = {doc_id: self.cosine_similarity(query_vector, doc_id) for doc_id in self.docs}
        ranked_docs = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(doc, score) for doc, score in ranked_docs if score > 0][:10]

# Initialize the model
vsm = VectorSpaceModel()

# User input loop for queries
while True:
    query = input("Enter your search query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    print("\nTop documents (Name, Score):")
    ranked_docs = vsm.rank_documents(query)
    for doc, score in ranked_docs:
        print(f"('{doc}.txt', {score:.17f})")
