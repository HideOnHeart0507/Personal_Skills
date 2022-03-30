import requests
from bs4 import BeautifulSoup
# 拿到页面源代码->用bs4解析

url = "http://www.xinfadi.com.cn/index.html"

resp = requests.get(url)


page = BeautifulSoup(resp.text, "html.parser")
# attrs={} = 属性=“”
table = page.find("table", class_="tbl-body")
trs = page.find_all("tr")[1:]

for tr in trs:
    tds = tr.find_all("td")
    print(tds)
#find 找一个 find_all找所有