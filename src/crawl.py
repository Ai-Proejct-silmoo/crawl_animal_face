from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import os
import urllib.request

print("원하는 인물을 입력하세요.\n")

person: str = input()

def crawl(title: str) :
    url = f'https://www.google.com/search?q={title}&sxsrf=APwXEdcjaKINAEK7K8Jn-it_E_QpBTs7XA:1684972588821&source=lnms&tbm=isch&sa=X&ved=2ahUKEwidrfOElI__AhWZcGwGHcUmCmYQ_AUoAXoECAEQAw&biw=1680&bih=946&dpr=2'
    options = Options()
    options.headless = True
    options.add_argument('window-size=1920x1080')
    options.add_argument('disable-gpu')

    driver = webdriver.Chrome('./chromedriver', options=options)

    driver.get(url)

    return driver.page_source

def get_image_url(html):
    soup = bs(html, 'lxml')
    class_name: str = "rg_i Q4LuWd"
    images_url = []
    image_element = soup.find_all('img', class_=class_name)
    
    for eleemnt in image_element:
        src = eleemnt.get('src')
        if src and src.startswith('http'):
            images_url.append(src)

    return images_url

def download_image(data, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, url in enumerate(data): 
        response = requests.get(url)
        file_path = os.path.join(path, f'image{idx}.jpg')
        urllib.request.urlretrieve(url, file_path)

html = crawl(person)

data = get_image_url(html)

#원하는 경로 복붙
download_image(data, "")