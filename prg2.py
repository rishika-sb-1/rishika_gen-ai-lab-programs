#!pip install gensim numpy matplotlib scikit-learn
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
print("Loading pre-trained word vectors...")
word_vectors = api.load("word2vec-google-news-300")

def explore_word_relationships(word1, word2, word3):
    try:
        result_vector = word_vectors[word1] - word_vectors[word2] + word_vectors[word3]
        similar_words = word_vectors.similar_by_vector(result_vector, topn=10)
        # Exclude input words from the results
        input_words = {word1, word2, word3}
        filtered_words = [
            (word, similarity)
            for word, similarity in similar_words if word not in input_words]
        print(f"\nWord Relationship: {word1} - {word2} + {word3}")
        print("Most similar words to the result (excluding input words):")
        for word, similarity in filtered_words[:5]:
            print(f"{word}: {similarity:.4f}")
        return filtered_words 
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")
        return []
        
def visualize_word_embeddings(words):
    vectors = np.array([word_vectors[word] for word in words])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1], marker='o', color='blue')
        plt.text(reduced[i, 0] + 0.02, reduced[i, 1] + 0.02, word, fontsize=12)
    plt.title(f"Word Embeddings Visualization using PCA")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
 
words_to_explore = ["king", "man", "woman", "queen", "prince", "princess", "royal", "throne"]
visualize_word_embeddings(words_to_explore)
domain_words = [
    "computer", "software", "hardware", "algorithm", "data",
    "network", "programming", "machine", "learning", "artificial"]
visualize_word_embeddings(domain_words)
def generate_similar_words(word):
    try:
        similar_words = word_vectors.most_similar(word, topn=5)
        print(f"\nTop 5 semantically similar words to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")
generate_similar_words("computer")
generate_similar_words("learning")