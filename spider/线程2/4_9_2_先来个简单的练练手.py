"""
流程:
    1. 拿到548121-1-1.html的页面源代码
    2. 从源代码中提取到m3u8的url
    3. 下载m3u8
    4. 读取m3u8文件, 下载视频
    5. 合并视频
"""
import requests
import re


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36"
}

obj = re.compile(r"url: '(?P<url>.*?)',", re.S)  # 用来提取m3u8的url地址

url = "https://www.91kanju.com/vod-play/54812-1-1.html"

resp = requests.get(url, headers=headers)
m3u8_url = obj.search(resp.text).group("url")  # 拿到m3u8的地址

# print(m3u8_url)
resp.close()

# 下载m3u8文件
resp2 = requests.get(m3u8_url, headers=headers)

# with open("哲仁王后.m3u8", mode="wb") as f:
#     f.write(resp2.content)
#
# resp2.close()
# print("下载完毕")


# 解析m3u8文件
# n = 1
# with open("哲仁王后.m3u8", mode="r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()  # 先去掉空格, 空白, 换行符
#         if line.startswith("#"):  # 如果以#开头. 我不要
#             continue
#
#         # 下载视频片段
#         resp3 = requests.get(line)
#         f = open(f"video/{n}.ts", mode="wb")
#         f.write(resp3.content)
#         f.close()
#         resp3.close()
#         n += 1
#         print("完成了1个")

