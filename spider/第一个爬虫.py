#爬虫：通过编写程序来获得互联网的资源
#百度
#需求 用程序模拟浏览器，输入一个网址，从该网址中获取到资源、neirong

from urllib.request import urlopen

url = "http://www.baidu.com"
response = urlopen(url)

with open("../mybaidu.html", mode="w", encoding="utf-8") as f:
    f.write(response.read().decode("utf-8")) #读取页面源代码
f.close()
print("over!")