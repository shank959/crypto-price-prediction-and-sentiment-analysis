from transformers import pipeline
import re

def preprocess_text(text):
    # Keep periods for sentence splitting
    text = text.lower()
    text = re.sub(r'[^a-z\s\.]', '', text)  # Retain periods
    return text

def sentiment_scores(articles):

    # Initialize the pipeline with truncation parameters
    pipe = pipeline(
        "text-classification",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        device=-1
    )

    sentiment_table = {}

    for title, body in articles.items():
        try:
            full_body = ' '.join(body)
            text = preprocess_text(f"{title}. {full_body}")

            # Split text into sentences
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings

            if not sentences:
                print(f"No sentences found for article '{title}'. Skipping.")
                continue

            results = []
            batch_size = 16  # Adjust based on your system's capacity
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                try:
                    # Call the pipeline with truncation
                    batch_results = pipe(batch, truncation=True, max_length=512)
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch starting at sentence {i} in article '{title}': {e}")
                    continue

            if not results:
                print(f"No sentiment results for article '{title}'.")
                continue

            # Determine Final Label and Score
            sentiment_mapping = {
                'positive': 1,
                'neutral': 0,
                'negative': -1
            }

            total_weighted_score = 0
            total_confidence = 0
            for result in results:
                label = result['label'].lower()
                sentiment_score = sentiment_mapping.get(label)
                if sentiment_score is None:
                    print(f"Unknown sentiment label '{result['label']}' in article '{title}'. Skipping.")
                    continue
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

        except Exception as e:
            print(f"Error processing article '{title}': {e}")
            continue

    return sentiment_table



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


