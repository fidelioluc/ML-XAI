from sklearn.datasets import fetch_20newsgroups
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- Sentence Counting ----------
def count_sentences(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    return len(list(parser.document.sentences))

# ---------- Summarization ----------
def get_summary(text, ratio=0.33):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        num_sentences = max(1, int(len(parser.document.sentences) * ratio))
        summary_sentences = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary_sentences)
    except Exception:
        return None

# ---------- Pipeline ----------
def summarize_filtered_20newsgroups(n=1000, min_sentences=6):
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    raw_texts = data.data[:n]
    targets = data.target[:n]
    target_names = [data.target_names[i] for i in targets]

    results = []

    for text, target, target_name in tqdm(zip(raw_texts, targets, target_names), total=n, desc="Filtering & Summarizing"):
        cleaned = clean_text(text)
        try:
            if count_sentences(cleaned) < min_sentences:
                continue
        except Exception:
            continue

        summary_66 = get_summary(cleaned, ratio=0.66)
        summary_33 = get_summary(cleaned, ratio=0.33)

        if summary_66 and summary_33:
            results.append({
                'target': target,
                'target_name': target_name,
                'full_text': cleaned,
                'summary_66': summary_66,
                'summary_33': summary_33
            })

    return pd.DataFrame(results)


# ---------- Run Script ----------
if __name__ == "__main__":
    df = summarize_filtered_20newsgroups(n=1000, min_sentences=6)
    df.to_csv("20newsgroup_summaries.csv", index=False)
    print(f"✅ Saved {len(df)} summarized rows to 20newsgroup_summaries.csv")




