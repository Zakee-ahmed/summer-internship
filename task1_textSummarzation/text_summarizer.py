# text_summarizer.py

from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if _name_ == "_main_":
    print("===== TEXT SUMMARIZER =====")
    input_text = input("Paste the article/text to summarize:\n")
    print("\nSUMMARY:")
    summary = summarize_text(input_text)
    print(summary)
