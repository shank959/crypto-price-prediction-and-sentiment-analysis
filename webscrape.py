from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import requests
import re
import time
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException, NoSuchElementException


def scrape_news(coin):

    # Initialize the Driver
    service = Service(executable_path='./chromedriver')
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=service, options=options)

    url = "https://coinmarketcap.com/currencies/" + coin + "/#News"
    try:
        driver.get(url)

        driver.implicitly_wait(5)  #! REMOVE WHEN WIFI DECENT
        element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "section-coin-news"))).find_element(By.TAG_NAME, 'section')




        driver.execute_script("window.scrollTo(0,1589.5)")
        element = driver.find_element(By.CSS_SELECTOR, ".sc-4c05d6ef-0:nth-child(3) > a > .sc-4c05d6ef-0 > .sc-4c05d6ef-0 > .sc-4c05d6ef-0 > .sc-65e7f566-0")
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        element = driver.find_element(By.CSS_SELECTOR, "body")
        actions = ActionChains(driver)
        actions.move_to_element(element, 0, 0).perform()
        driver.find_element(By.CSS_SELECTOR, ".ckzuDZ > .BaseButton_btnContentWrapper__vm2Ue > .sc-65e7f566-0").click()
        driver.find_element(By.CSS_SELECTOR, ".ckzuDZ > .BaseButton_btnContentWrapper__vm2Ue > .sc-65e7f566-0").click()
        driver.find_element(By.CSS_SELECTOR, ".ckzuDZ > .BaseButton_btnContentWrapper__vm2Ue > .sc-65e7f566-0").click()
        element = driver.find_element(By.CSS_SELECTOR, ".ckzuDZ > .BaseButton_btnContentWrapper__vm2Ue > .sc-65e7f566-0")
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        element = driver.find_element(By.CSS_SELECTOR, "body")
        actions = ActionChains(driver)
        actions.move_to_element(element, 0, 0).perform()

       



        tags = element.find_elements(By.TAG_NAME, 'a')

        articles = {}
        for tag in tags:
            title = tag.find_element(By.TAG_NAME, 'h5').text
            articles[title] = tag.get_attribute('href')
        driver.quit()
        return articles

    except:
        print("Error: Cannot scrape coin news")
        return None



def retrieve_article(url):
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

    return article_text


if __name__ == "__main__":
    hrefs = scrape_news('solana')
    if not hrefs:
        print("Error: No News Articles Found")
        exit()
    else:
        print(hrefs)



