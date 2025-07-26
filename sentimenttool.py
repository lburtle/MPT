import re
import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import torch
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize
# import matplotlib.pyplot as plt
import requests
from newspaper import Article
import os
import json

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab resource...")
    nltk.download('punkt_tab')

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def get_articles(query, api_key, num_articles=100, use_full_text=True):
    articles = []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        from_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
        response = newsapi.get_everything(
        q=f"{query} (earnings OR revenue OR profit OR loss OR investor OR valuation OR shares OR dividend OR analyst OR shareholder OR bullish OR bearish OR equity) -wildfire -deal -sale",
        language='en',
        sort_by='relevancy',
        page_size=min(num_articles, 100),
        from_param=from_date
        )

        for article in response['articles']:
            text = (article.get('title', '') + ' ' +
                    article.get('description', '') + ' ' +
                    article.get('content', '')).strip()
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 50:
                if use_full_text:
                    try:
                        news_article = Article(article.get('url', ''))
                        news_article.download()
                        news_article.parse()
                        full_text = clean_text(news_article.text)
                        if full_text and len(full_text) > len(cleaned_text):
                            cleaned_text = full_text
                    except Exception as e:
                        print(f"Failed to fetch full text for {article.get('title', 'No Title')}: {e}")
                articles.append({
                    'title': article.get('title', 'No Title'),
                    'text': cleaned_text,
                    'url': article.get('url', '')
                })
    except Exception as e:
        print(f"Error fetching articles from NewsAPI: {e}")
    return articles

def initialize_classifier():
    try:
        if not torch.__version__:
            raise ImportError("PyTorch is not installed.")
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return classifier
    except Exception as e:
        print(f"Error initializing FinBERT classifier: {e}")
        print("Please ensure PyTorch and transformers are installed")
        return None
    
def analyze_sentiment(text, classifier, financial_keywords):
    if not classifier:
        return {'label': 'neutral', 'score': 0, 'sentences': []}
    try:
        sentences = sent_tokenize(text)
        financial_sentences = [s for s in sentences if any(kw in s.lower() for kw in financial_keywords)]
        if not financial_sentences:
            return {'label': 'neutral', 'score': 0, 'sentences': []}
        
        scores = []
        labels = []
        analyzed_sentences = []
        for sentence in financial_sentences[:5]:
            sentence = sentence[:512] if len(sentence) > 512 else sentence
            result = classifier(sentence)[0]
            labels.append(result['label'])
            
            if result['score'] > 0.7:
                score = result['score'] * {'positive': 1, 'negative': -1, 'neutral': 0}[result['label']]
                scores.append(score)
            analyzed_sentences.append({'sentence': sentence, 'label': result['label'], 'score': result['score']})

        if scores:
            non_neutral = [s for s in scores if s != 0]
            if non_neutral:
                avg_score = sum(non_neutral) / len(non_neutral)
                label = 'positive' if avg_score > 0 else 'negative'
            else:
                avg_score = 0
                label = 'neutral'
            return {'label': label, 'score': avg_score, 'sentences': analyzed_sentences}
        return {'label': 'neutral', 'score': 0, 'sentences': analyzed_sentences}
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {'label': 'neutral', 'score': 0, 'sentences': []}

# def plot_sentiment_distribution(sentiment_counts, query):
#     labels = ['Positive', 'Negative', 'Neutral']
#     counts = [sentiment_counts.get('positive', 0), sentiment_counts.get('Negative', 0), sentiment_counts.get('neutral', 0)]

#     fig, ax = plt.subplots()
#     ax.bar(labels, counts, color=['#00FF00', '#FF0000', '#808080'])
#     ax.set_ylabel('Number of Articles')
#     ax.set_xlabel('Sentiment')
#     ax.set_title(f'Sentiment Distribution for {query}')
    
#     # Save plot to file
#     plt.savefig('sentiment_chart.png', format='png', bbox_inches='tight')

def sentiment_analysis(query, api_key, num_articles=100, use_full_text=False):
    print(f"Fetching articles for: {query}")
    classifier = initialize_classifier()
    if not classifier:
        return "Cannot proceed without sentiment analysis due to missing FinBERT classifier."
    
    articles = get_articles(query, api_key, num_articles, use_full_text)

    if not articles:
        return "No articles found for the query"
    
    results = []
    skipped = 0
    financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'investor', 'valuation', 'shares',
            'dividend', 'analyst', 'shareholder', 'bullish', 'bearish', 'equity'
        ]

    for article in articles:
        text_lower = (article['title'] + ' ' + article['text']).lower()
        found_keywords = [kw for kw in financial_keywords if kw in text_lower]
        if not found_keywords:
            print(f"Skipped article (no financial keywords): {article['title']} (Found: {found_keywords})")
            skipped += 1
            continue
        sentiment = analyze_sentiment(article['text'], classifier, financial_keywords)
        results.append({
            'title': article['title'],
            'url': article['url'],
            'sentiment_label': sentiment['label'],
            'sentiment_score': sentiment['score'],
            'keywords': found_keywords,
            'text_snippet': article['text'][:200] + '...' if len(article['text']) > 200 else article ['text'],
            'financial_sentences': sentiment['sentences']
        })

    if not results:
        return f"No financially relevant articles found for the query. Skipped {skipped} articles."
    
    df = pd.DataFrame(results)

    avg_sentiment = df['sentiment_score'].mean()

    if avg_sentiment > 0.1:
        overall_sentiment = 'Positive'
    elif avg_sentiment < -0.1:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'

    sentiment_counts = df['sentiment_label'].value_counts().to_dict()

    # plot_sentiment_distribution(sentiment_counts, query)

    output = f"\nFinancial Sentiment Analysis Results for '{query}':\n"
    output += f"Number of articles fetched: {len(articles)}\n"
    output += f"Number of articles analyzed: {len(df)}\n"
    output += f"Number of articles skipped (non-financial): {skipped}\n"
    output += f"Average sentiment score: {avg_sentiment:.3f}\n"
    output += f"Overall investment sentiment: {overall_sentiment}\n\n"
    # output += "Article Details:\n"
    # for _, row in df.iterrows():
    #     output += f"Title: {row['title']}\n"
    #     output += f"URL: {row['url']}\n"
    #     output += f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_score']:.3f})\n"
    #     output += f"Financial Keywords: {row['keywords']}\n"
    #     output += f"Text Snippet: {row['text_snippet']}\n"
    #     if row['financial_sentences']:
    #         output += "Financial Sentences Analyzed:\n"
    #         for sent in row['financial_sentences']:
    #             output += f"  - {sent['sentence']} (Sentiment: {sent['label']}, Score: {sent['score']:.3f})\n"
    #     else:
    #         output += "Financial Sentences Analyzed: None\n"
    #     output += "\n"
    
    # output += "\nSentiment Distribution Chart displayed and saved as 'sentiment_chart.png' in the current directory.\n"
    # output += "If the chart did not display, ensure your environment supports matplotlib GUI (e.g., local machine or Jupyter notebook).\n"
    

    result = {
    "query": query,
    "num_fetched": len(articles),
    "num_analyzed": len(df),
    "num_skipped": skipped,
    "avg_sentiment": avg_sentiment,
    "overall_sentiment": overall_sentiment # This should be a list of dicts for each article
    }

    return output, result

if __name__ == "__main__":
    api_key = os.getenv("API_KEY")
    query = "NVIDIA"
    output, result = sentiment_analysis(query, api_key, num_articles=100)
    print(output)
    if isinstance(result, dict) and result:
        with open("/output/result.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        with open("/output/result.json", "w") as f:
            f.write(output)