from transformers import pipeline
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def sentiment_scores(articles):

    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=-1)

    sentiment_table = {}

    try:
        for title, body in articles.items():
            text = preprocess_text(f"{title}. {body}")

            sentences = text.split('.')
            results = pipe(sentences)

            # # Display the results
            # for sentence, result in zip(sentences, results):
            #     print(f"Sentiment: {result['label']}, Score: {result['score']}\n")  

            # Determine Final Label and Score
            sentiment_mapping = {
                'positive': 1,
                'neutral': 0,
                'negative': -1
            }

            total_weighted_score = 0
            total_confidence = 0
            for result in results:
                sentiment_score = sentiment_mapping[result['label']]
                confidence_score = result['score']
                weighted_score = sentiment_score * confidence_score
                total_weighted_score += weighted_score
                total_confidence += confidence_score

            average_weighted_score = (total_weighted_score / total_confidence) if total_confidence != 0 else 0

            neutral_threshold_upper = 0.1   
            neutral_threshold_lower = -0.1

            if average_weighted_score > neutral_threshold_upper:
                final_sentiment = 1
            elif average_weighted_score < neutral_threshold_lower:
                final_sentiment = -1
            else:
                final_sentiment = 0

            sentiment_table[title] = final_sentiment

        return sentiment_table
    
    except:
        print("Error running sentiment analysis")
        return {}


def sentiment_count(sentiment_table):
    positive, neutral, negative = 0, 0, 0
    for sentiment in sentiment_table.values():
        if sentiment == 1:
            positive += 1
        elif sentiment == -1:
            negative += 1
        else:
            neutral += 1
    return [positive, neutral, negative]


