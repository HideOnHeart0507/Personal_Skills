# 拿页面源代码->解析

import requests
from lxml import etree

url = "https://supei.zbj.com/search/service?kw=saas&r=2"
resp = requests.get(url)
print(resp.text)
# print(resp.text)

html = etree.HTML(resp.text)
divs= html.xpath("/html/body/div[2]/div/div/div[2]/div/div[2]/div[3]/div[1]/div")
for div in divs:
    price = div.xpath("./div[2]/div/span[1]/text()")
    print(div.xpath("./text()"))


price = html.xpath("/html/body/div[2]/div/div/div[2]/div/div[2]/div[3]/div[1]/div[1]/div[2]/div/span[1]/text()")
print(price)