# Program 6: Sentiment Analysis Using Hugging Face

# !pip install transformers ipywidgets

from transformers import pipeline

print("Loading Sentiment Analysis Model...")
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']

    print(f"\nInput Text: {text}")
    print(f"Sentiment: {label} (Confidence: {score:.4f})\n")

    return result

customer_reviews = [
    "The product is amazing! I love it so much.",
    "I'm very disappointed. The service was terrible.",
    "It was an average experience, nothing special.",
    "Absolutely fantastic quality! Highly recommended.",
    "Not great, but not the worst either."
]

print("\nCustomer Sentiment Analysis Results:")

for review in customer_reviews:
    analyze_sentiment(review)
