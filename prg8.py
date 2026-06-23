from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=100, min_length=30):
    """
    Summarizes the given text using a pre-trained model.

    Parameters:
        text (str): The input passage to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    if len(text.split()) < min_length:  # Avoid issues with very short text
        return "Text is too short to summarize."

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example input passage
text = """Artificial intelligence (AI) is transforming various industries by automating tasks,
improving efficiency, and enabling new capabilities. In healthcare, AI assists doctors in diagnosing diseases,
predicting patient outcomes, and personalizing treatment plans. In finance, AI-driven algorithms detect fraudulent
activities and optimize investment strategies. In transportation, autonomous vehicles are becoming a reality,
reducing human error and improving safety. As AI continues to advance, ethical considerations such as bias, privacy,
and job displacement must be addressed to ensure responsible development and deployment."""

# Get the summary
summary_result = summarize_text(text)

# Print the summarized text
print("Summary:", summary_result)