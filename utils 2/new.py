from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import re
import time



def scrape_news():

    # Initialize the Driver
    service = Service(executable_path='./chromedriver')
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=service, options=options)

    url = "https://cryptoslate.com/?s=avalanche"

    driver.get(url)
    driver.implicitly_wait(5)
    driver.find_element(By.LINK_TEXT, "Load more articles").click()
    driver.implicitly_wait(5)
    driver.find_element(By.LINK_TEXT, "Load more articles").click()
    driver.implicitly_wait(5)



    # table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.search-objects.list-feed.news-search-results"))).find_element(By.TAG_NAME, 'section')


    
   




if __name__ == "__main__":
    articles = scrape_news()





