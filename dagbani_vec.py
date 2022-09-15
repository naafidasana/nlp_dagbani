from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_doc(filename):
    # return a list of sentences from the document passed.
    with open(filename, 'rt', encoding='utf-8') as file:
        txt = file.read().split("\n")
    return txt

path = r"/dt3.txt"

sentences = read_doc(path)

# split each sentence into a list of words.
sentences = [sent.split() for sent in sentences]

# Train Word2Vec model
model = Word2Vec(sentences, min_count=5, alpha=0.5 )
# Save model
model.save('dagbani_word_model.bin')

# Load model.
pt_model = Word2Vec.load('dagbani_word_model.bin')

# Perform routines for visualisation
X = pt_model[pt_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# Create scatter plot for projection.
plt.scatter(result[:, 0], result[:, 1])
words = list(set(pt_model.wv.vocab))
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))


plt.title("Visualisation of Dagbani Word Embeddings using PCA")
plt.show()

# save plot
plt.savefig(r"F:\Projects\dagbani_nlp\dagbani_word_vectors.png")
  