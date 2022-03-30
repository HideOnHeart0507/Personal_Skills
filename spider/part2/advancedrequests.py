# 登录 -》得到cookie
# 带着cookie去请求书架url -》 书架上内容

# 必须把上面两个连起来操作
# 使用session请求 一连串请求

import requests
# cookie
# session = requests.session()

#登录

# data = {
#     "loginName": "18614075987",
#     "password": "q6035945"
# }
#
# url = "https://passport.17k.com/ck/user/login"
# resp = session.post(url, data=data)
# print(resp.cookies)
# 接下来继续session 可以不用header加cookie

# 防盗链
# 拿到contID， 拿到video里面的json 找到其中视频url srcurl里面调整
urL = 'https://www.pearvideo.com/video_1753755'
contID = urL.split('_')[1]
headers ={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
    "Referer": urL
#     防盗链↑ 溯源， 当前请求的上一级
}
status = f'https://www.pearvideo.com/videoStatus.jsp?contId={contID}&mrd=0.3114160589587389'

resp = requests.get(status, headers=headers)

systemTime = resp.json()['systemTime']

url1=resp.json()['videoInfo']['videos']['srcUrl']

url1 =url1.replace(systemTime, f"cont-{contID}")
print(url1)
#下载视频
with open('example.mp4', mode='wb') as f:
    f.write(requests.get(url1).content)
f.close()
resp.close()

# 代理
proxies = {
    "https" : "https://218.60.8"
}

resp = requests.get(url, proxies=proxies)