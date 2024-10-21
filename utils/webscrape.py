from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import re



def scrape_news(coin_name, num_pages=1):

    # Initialize the Driver
    service = Service(executable_path='./chromedriver')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=service, options=options)

    url = "https://cryptoslate.com/news/" + coin_name
    
    def scrape_page(page_num):
        driver.get(f"{url}/page/{page_num}/")
        # driver.implicitly_wait(5)  #! REMOVE WHEN WIFI DECENT
        sub_tags = []
        if page_num == 1:
            main_tag = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.list-feed.grid"))).find_element(By.TAG_NAME, 'a')
            sub_tags.append(main_tag)
        feed_tags = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "section.list-feed.grid"))).find_elements(By.TAG_NAME, 'a')
        sub_tags.extend(feed_tags)
        return sub_tags
    
    hrefs = {}
    for page_num in range(1, num_pages + 1):
        try:
            sub_tags = scrape_page(page_num)
        except:
            print(f"Error: Cannot scrape page {page_num}")
            continue
        for sub_tag in sub_tags:
            title = sub_tag.get_attribute('title')
            hrefs[title] = sub_tag.get_attribute('href')

    driver.quit()
    return hrefs




def retrieve_articles(hrefs):
    try:
        articles = {}
        for title, url in hrefs.items():
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            article_text = []
            article_body = soup.find('article')
            if not article_body:
                article_body = soup.find('div', class_=re.compile('main|content|article|body', re.I))

            if article_body:
                paragraphs = article_body.find_all('p')
                for paragraph in paragraphs:
                    if paragraph.string:
                        article_text.append(paragraph.string)

            if not article_text:
                # Fallback: Get all <p> tags in the document
                paragraphs = soup.find_all('p')
                for paragraph in paragraphs:
                    if paragraph.string:
                        article_text.append(paragraph.string)
            
            if not article_text:
                article_text = "Not Found"

            articles[title] = article_text

        return articles
    except:
        return None




if __name__ == "__main__":
    articles = scrape_news('solana', 3)
    if not articles:
        print("Error: No News Articles Found")
        exit()
    else:
        for key,value in articles.items():
            print(f"{key}: {value}")




