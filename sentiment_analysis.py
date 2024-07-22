from transformers import pipeline


placeholder_text = """
The solana coin performed exceptionally well on yesterdays market, rising 12 percent by the days close. Investors are optimistic about the future. However, there are some concerns about the sustainability of this growth. This growth may be at its peak as of today, and may experience an upcoming market correction for its currently overvalued price.
"""

def sentiment_score(text):

    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=-1)

    sentences = text.split('.')
    sentence_count = len(sentences)
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

    if  total_confidence != 0:
        average_weighted_score = total_weighted_score / total_confidence
    else:
        average_weighted_score = 0
    average_confidence = total_confidence / sentence_count

    if average_weighted_score > 0:
        final_sentiment = 1
    elif average_weighted_score < 0:
        final_sentiment = -1
    else:
        final_sentiment = 0

    return final_sentiment
    # print(f"The Average Weighted Score is {average_weighted_score}")


