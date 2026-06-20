# Program 7: Text Summarization Using Hugging Face

# !pip install transformers ipywidgets

from transformers import pipeline

print("Loading Summarization Model (BART)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=None, min_length=None):
    text = " ".join(text.split())

    if not max_length:
        max_length = min(len(text) // 3, 150)

    if not min_length:
        min_length = max(30, max_length // 3)

    summary_1 = summarizer(text, max_length=150, min_length=30, do_sample=False)
    summary_2 = summarizer(text, max_length=150, min_length=30, do_sample=True, temperature=0.9)
    summary_3 = summarizer(text, max_length=150, min_length=30, do_sample=False, num_beams=5)
    summary_4 = summarizer(text, max_length=150, min_length=30, do_sample=True, top_k=50, top_p=0.95)

    print("\nOriginal Text:")
    print(text)

    print("\nSummarized Text:")

    print("\nDefault:")
    print(summary_1[0]['summary_text'])

    print("\nHigh randomness:")
    print(summary_2[0]['summary_text'])

    print("\nConservative:")
    print(summary_3[0]['summary_text'])

    print("\nDiverse Sampling:")
    print(summary_4[0]['summary_text'])

long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines capable of mimicking human cognitive functions such as learning, problem-solving, and decision-making. In recent years, AI has significantly impacted various industries, including healthcare, finance, education, and entertainment. AI-powered applications, such as chatbots, self-driving cars, and recommendation systems, have transformed the way we interact with technology. Machine learning and deep learning, subsets of AI, enable systems to learn from data and improve over time without explicit programming. However, AI also poses ethical challenges, such as bias in decision-making and concerns over job displacement. As AI technology continues to advance, it is crucial to balance innovation with ethical considerations to ensure its responsible development and deployment.
"""

summarize_text(long_text)
