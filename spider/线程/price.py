from concurrent.futures.thread import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

def download_one_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.encoding='gbk'
    page = BeautifulSoup(resp.text, "html.parser")
    table = page.find("div", class_="showtxt")
    str = table.text
    print(table.text)
def download_main_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = 'gbk'
    page = BeautifulSoup(resp.text, "html.parser")
    dl = page.find("div", class_='listmain')
    href_list = dl.find_all("a")
    counter = 1
    for a in href_list:
        href = a.get("href")
        url = "https://www.bqkan8.com" + href
        resp = requests.get(url, headers=headers)
        resp.encoding = 'gbk'
        page = BeautifulSoup(resp.text, "html.parser")
        dl = page.find("div", class_='showtxt')
        with open(f"第{counter}章" , mode="w", encoding='utf-8') as f:
            f.write(dl.text)
        f.close()
        break

if __name__ == '__main__':
    url = "https://www.bqkan8.com/1_1496/"
    # download_one_page(url)
    with ThreadPoolExecutor(50) as t:
        for i in range(1,200):
            t.submit(download_one_page(),url)
    download_main_page(url)