import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

custom_stopwords = [
    "bitcoin",
    "ethereum",
    "solana",
    "dogecoin",
    "cardano",
    "tether",
    "tron",
    "polkadot",
    "chainlink",
    "polygon",
    "litecoin"
    "crypto"
    "news"
    "etf"
]

def create_wordcloud_dictionary(article_bodies, sentiment_dictionary):

    sentiment_body_list = {}
    for title, body in article_bodies.items():
        full_body = ' '.join(body)
        sentiment_body_list[full_body] = sentiment_dictionary[title]
    
    return sentiment_body_list


def generate_word_cloud(sentiment_body_list, sentiment_label):
    # Filter the articles based on the sentiment label
    filtered_bodies = [body for body, sentiment in sentiment_body_list.items() if sentiment == sentiment_label]
    
    if not filtered_bodies:
        return None  # Return None if there are no articles for the sentiment

    # Combine all the article bodies into a single string
    text = ' '.join(filtered_bodies)

    # Create the set of stopwords, including the custom words
    stopwords = set(STOPWORDS)
    stopwords.update(custom_stopwords)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        colormap='viridis',
        max_words=100
    ).generate(text)

    return wordcloud