# Program 1: Explore Pre-trained Word Vectors

# Install necessary libraries
# !pip install gensim numpy

import gensim.downloader as api
import numpy as np
from numpy.linalg import norm

print("Loading pre-trained word vectors...")
word_vectors = api.load("word2vec-google-news-300")

def explore_word_relationships(word1, word2, word3):
    try:
        vec1 = word_vectors[word1]
        vec2 = word_vectors[word2]
        vec3 = word_vectors[word3]

        result_vector = vec1 - vec2 + vec3
        similar_words = word_vectors.similar_by_vector(result_vector, topn=10)

        input_words = {word1, word2, word3}
        filtered_words = [(word, similarity) for word, similarity in similar_words if word not in input_words]

        print(f"\nWord Relationship: {word1} - {word2} + {word3}")
        print("Most similar words to the result (excluding input words):")
        for word, similarity in filtered_words[:5]:
            print(f"{word}: {similarity:.4f}")

    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

explore_word_relationships("king", "man", "woman")
explore_word_relationships("paris", "france", "germany")
explore_word_relationships("apple", "fruit", "carrot")

def analyze_similarity(word1, word2):
    try:
        similarity = word_vectors.similarity(word1, word2)
        print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

analyze_similarity("cat", "dog")
analyze_similarity("computer", "keyboard")
analyze_similarity("music", "art")

def find_most_similar(word):
    try:
        similar_words = word_vectors.most_similar(word, topn=5)
        print(f"\nMost similar words to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

find_most_similar("happy")
find_most_similar("sad")
find_most_similar("technology")
