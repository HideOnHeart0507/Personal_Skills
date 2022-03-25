import requests

url = 'https://fanyi.baidu.com/sug'
s = input("输入你要翻译的英文")
dat = {
    "kw": s
}

resp = requests.post(url, data=dat)
print(resp.json()) #直接处理成json
resp.close()