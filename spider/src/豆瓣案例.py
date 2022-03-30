import requests


url = "https://movie.douban.com/j/chart/top_list?"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"
}
#封装参数
param = {
    "type":"24",
    "interval_id":"100:90",
    "action":"",
    "start":0,
    "limit":20
}

req = requests.get(url, params = param, headers=headers)
print(req.json())
req.close()