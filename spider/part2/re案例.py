#页面源代码->正则提取

import requests
import re
import csv

headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"
}
n = 0
for i in range(10):
    url = f"https://movie.douban.com/top250?start={n}&filter="
    n += 25
    resp = requests.get(url, headers=headers)
    page_html = resp.text

    obj = re.compile(r'<li>.*?<div class="item">.*?<span class="title">(?P<title>.*?)'
                     r'</span>.*?<p class="">.*?<br>(?P<year>.*?)&nbsp'
                     r'.*?<span class="rating_num" property="v:average">(?P<score>.*?)</span>'
                     r'.*?</span>.*?<span>(?P<number>.*?)</span>', re.S)
    result = obj.finditer(page_html)

    f = open("douban.csv", mode="a", encoding="UTF-8")
    csvwriter = csv.writer(f)

    for i in result:
        dic = i.groupdict()
        dic['year'] = dic['year'].strip()
        csvwriter.writerow(dic.values())

    f.close()
resp.close()
