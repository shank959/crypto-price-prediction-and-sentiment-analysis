from sentiment_analysis import sentiment_score
from webscrape import retrieve_article, scrape_news

def main():
    hrefs = scrape_news('cardano')
    if not hrefs:
        print("Error: No News Articles Found")
    for title, href in hrefs.items():
        print(f"{title}: {href}")
    exit()

    articles = {}
    for title, href in hrefs.items():
        text = retrieve_article(href)
        articles[title] = text

    sentiment_table = {}
    for key, value in articles.items():
        text = f"{key} {value}"
        sentiment = sentiment_score(text)
        label = 'Positive' if sentiment == 1 else 'Negative' if sentiment == -1 else 'Neutral'
        # print(f"{key}: {label} ({confidence.round(3)})")
        sentiment_table[key] = label

    print(sentiment_table)


if __name__ == "__main__":
    main()
