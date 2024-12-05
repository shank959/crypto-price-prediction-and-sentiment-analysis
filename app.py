import streamlit as st
import matplotlib.pyplot as plt
from utils.sentiment_analysis import sentiment_scores, sentiment_count
from utils.webscrape import scrape_news, retrieve_articles
from utils.helper_functions import create_pie_chart, create_table_df
from utils.prices import fetch_crypto_price, load_coin_list, get_coin_id_by_name
from utils.wordcloud import create_wordcloud_dictionary, generate_word_cloud


NUMBER_OF_PAGES_TO_SCRAPE = 3

# List of coin options
coin_options = [
    "Bitcoin",
    "Ethereum",
    "Solana",
    "Dogecoin",
    "Cardano",
    "Tether",
    "Tron",
    "Polkadot",
    "Chainlink",
    "Polygon",
    "Litecoin"
]

# Main function for the Streamlit app
def main():

    # Initializations
    coin_list = load_coin_list()

    st.title("Cryptocurrency Sentiment Analyzer")
    
    # Sidebar or Main Page: Coin selection
    st.subheader("Select a cryptocurrency:")
    coin = st.selectbox("Cryptocurrency", coin_options)
    
    # Handle invalid selection or other feedback
    st.write("Choose a cryptocurrency to analyze sentiment from recent news articles.")
    
    # Form submission to analyze sentiment
    if st.button("Analyze"):
        articles = scrape_news(coin_name=coin, num_pages=3)
        
        if not articles:
            st.error(f"No articles found for {coin}. Please try again.")
        else:
            # Perform sentiment analysis

            article_bodies = retrieve_articles(articles)
            sentiments = sentiment_scores(article_bodies)
            counts = sentiment_count(sentiments)
            count_percentages = [count / sum(counts) * 100 for count in counts]
            df = create_table_df(sentiments=sentiments)


            # Display crypto prices
            st.divider()
            price, change_1d = fetch_crypto_price(get_coin_id_by_name(coin, coin_list))
            col8, col9 = st.columns(2)
            with col8:
                st.metric("Current Market Price", str(price) + "USD")
            with col9:
                st.metric("1-Day Change", str(change_1d) + "%")
            st.divider()
            
            # Display sentiment analysis results
            st.write(f"### Sentiment Analysis Results for {coin}:")
            
            # Show sentiment counts
            st.write("#### Sentiment Counts")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", counts[0])
            with col2:
                st.metric("Neutral", counts[1])
            with col3:
                st.metric("Negative", counts[2])

            # Display the sentiment percentages
            st.write("#### Sentiment Percentage Distributions")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Positive", str(round(count_percentages[0], 2)) + "%")
            with col5:
                st.metric("Neutral", str(round(count_percentages[1], 2)) + "%")
            with col6:
                st.metric("Negative", str(round(count_percentages[2], 2)) + "%")
            
            # Display list of articles
            st.dataframe(df)  

            # Show pie chart
            st.write("### Sentiment Distribution")
            fig = create_pie_chart(counts)
            st.plotly_chart(fig)  # Render the pie chart

            # Generate wordcloud with a sentiment picker
            sentiment_body_list = create_wordcloud_dictionary(article_bodies, sentiments)
            st.write("### Word Cloud Based on Sentiment")

            # Generate the word cloud for the selected sentiment
            wordcloud_pos = generate_word_cloud(sentiment_body_list, 1)
            wordcloud_net = generate_word_cloud(sentiment_body_list, 1)
            wordcloud_neg = generate_word_cloud(sentiment_body_list, 1)

            st.write("#### Positive Sentiment Wordcloud")
            if wordcloud_pos:
                # Display the word cloud using matplotlib
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud_pos, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.write(f"No articles found for sentiment: Positive")

            st.write("#### Neutral Sentiment Wordcloud")
            if wordcloud_net:
                # Display the word cloud using matplotlib
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud_net, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.write(f"No articles found for sentiment: Neutral")
            
            st.write("#### Negative Sentiment Wordcloud")
            if wordcloud_neg:
                # Display the word cloud using matplotlib
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud_neg, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.write(f"No articles found for sentiment: Negative")



if __name__ == "__main__":
    main()
