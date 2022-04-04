import requests
import re

obj =re.compile(r"url:'(?P<url>.*?)',",re.S) #提取m3u8
url = "https://www.91kanju.com/vod-play/54812-1-1.html"
resp = requests.get(url)

m3u8_url = obj.search(resp.text).group("url")
print(m3u8_url)
