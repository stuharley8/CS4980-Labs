import numpy as np
import pandas as pd
import time
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')


# Function to remove stopwords
def remove_stopwords(sen):
    sen_new = ' '.join([i for i in sen if i not in stop_words])
    return sen_new


CHAPTER_PATH = input('Enter Chapter filename: ')
SUMMARY_LENGTH = 15

start = time.time()
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
print('Word embeddings created from Glove in', round(time.time() - start,2), 'seconds')

f = open(CHAPTER_PATH, 'r')
chapter = f.readline()  # Assumes entire chapter is contained in a single line
f.close()

start = time.time()
sentences = sent_tokenize(chapter)
# remove punctuation, numbers, and special characters. Make lowercase. Remove stopwords
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z ]", "")
clean_sentences = [s.lower() for s in clean_sentences]
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
print('Chapter sentences separated, cleaned, and stopwords removed in', round(time.time() - start,2), 'seconds')

start = time.time()
# Create Vector representation of sentences
sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)
print('Vector representation of cleaned sentences done in', round(time.time() - start,2), 'seconds')

start = time.time()
# Create Similarity Matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0,0]
print('Similarity Matrix of cleaned sentences created in', round(time.time() - start,2), 'seconds')

start = time.time()
# Apply PageRank Algorithm
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank_numpy(nx_graph)
print('PageRank Algorithm completed in', round(time.time() - start,2), 'seconds')

print('\nChapter Summary:')
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
for i in range(SUMMARY_LENGTH):
    print(ranked_sentences[i][1])
