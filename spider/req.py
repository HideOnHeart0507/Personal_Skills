import requests

query = input("你想搜索的内容")
url = f'https://www.sogou.com/web?query={query}'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"
}
resp = requests.get(url, headers=headers) #改headers处理反爬

#地址栏get

print(resp)
print(resp.text)
req.close()