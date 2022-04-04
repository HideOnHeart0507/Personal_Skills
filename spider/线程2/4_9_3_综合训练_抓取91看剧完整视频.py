"""
思路:
    1. 拿到主页面的页面源代码, 找到iframe
    2. 从iframe的页面源代码中拿到m3u8文件的地址
    3. 下载第一层m3u8文件 -> 下载第二层m3u8文件(视频存放路径)
    4. 下载视频
    5. 下载秘钥, 进行解密操作
    6. 合并所有ts文件为一个mp4文件
"""
import requests
from bs4 import BeautifulSoup
import re
import asyncio
import aiohttp
import aiofiles
from Crypto.Cipher import AES  # pycryptodome
import os


def get_iframe_src(url):
    resp = requests.get(url)
    main_page = BeautifulSoup(resp.text, "html.parser")
    src = main_page.find("iframe").get("src")
    return src
    # return "https://boba.52kuyun.com/share/xfPs9NPHvYGhNzFp"  # 为了测试


def get_first_m3u8_url(url):
    resp = requests.get(url)
    # print(resp.text)
    obj = re.compile(r'var main = "(?P<m3u8_url>.*?)"', re.S)
    m3u8_url = obj.search(resp.text).group("m3u8_url")
    # print(m3u8_url)
    return m3u8_url


def download_m3u8_file(url, name):
    resp = requests.get(url)
    with open(name, mode="wb") as f:
        f.write(resp.content)


async def download_ts(url, name, session):
    async with session.get(url) as resp:
        async with aiofiles.open(f"video2/{name}", mode="wb") as f:
            await f.write(await resp.content.read())  # 把下载到的内容写入到文件中
    print(f"{name}下载完毕")


async def aio_download(up_url):  # https://boba.52kuyun.com/20170906/Moh2l9zV/hls/
    tasks = []
    async with aiohttp.ClientSession() as session:  # 提前准备好session
        async with aiofiles.open("越狱第一季第一集_second_m3u8.txt", mode="r", encoding='utf-8') as f:
            async for line in f:
                if line.startswith("#"):
                    continue
                # line就是xxxxx.ts
                line = line.strip()  # 去掉没用的空格和换行
                # 拼接真正的ts路径
                ts_url = up_url + line
                task = asyncio.create_task(download_ts(ts_url, line, session))  # 创建任务
                tasks.append(task)

            await asyncio.wait(tasks)  # 等待任务结束


def get_key(url):
    resp = requests.get(url)
    return resp.text


async def dec_ts(name, key):
    aes = AES.new(key=key, IV=b"0000000000000000", mode=AES.MODE_CBC)
    async with aiofiles.open(f"video2/{name}", mode="rb") as f1,\
        aiofiles.open(f"video2/temp_{name}", mode="wb") as f2:

        bs = await f1.read()  # 从源文件读取内容
        await f2.write(aes.decrypt(bs))  # 把解密好的内容写入文件
    print(f"{name}处理完毕")


async def aio_dec(key):
    # 解密
    tasks = []
    async with aiofiles.open("越狱第一季第一集_second_m3u8.txt", mode="r", encoding="utf-8") as f:
        async for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            # 开始创建异步任务
            task = asyncio.create_task(dec_ts(line, key))
            tasks.append(task)
        await asyncio.wait(tasks)


def merge_ts():
    # mac: cat 1.ts 2.ts 3.ts > xxx.mp4
    # windows: copy /b 1.ts+2.ts+3.ts xxx.mp4
    lst = []
    with open("越狱第一季第一集_second_m3u8.txt", mode="r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            lst.append(f"video2/temp_{line}")

    s = " ".join(lst)  # 1.ts 2.ts 3.ts
    os.system(f"cat {s} > movie.mp4")
    print("搞定!")


def main(url):
    # 1. 拿到主页面的页面源代码, 找到iframe对应的url
    iframe_src = get_iframe_src(url)
    # 2. 拿到第一层的m3u8文件的下载地址
    first_m3u8_url = get_first_m3u8_url(iframe_src)
    # 拿到iframe的域名
    # "https://boba.52kuyun.com/share/xfPs9NPHvYGhNzFp"
    iframe_domain = iframe_src.split("/share")[0]
    # 拼接出真正的m3u8的下载路径
    first_m3u8_url = iframe_domain+first_m3u8_url
    # https://boba.52kuyun.com/20170906/Moh2l9zV/index.m3u8?sign=548ae366a075f0f9e7c76af215aa18e1
    # print(first_m3u8_url)
    # 3.1 下载第一层m3u8文件
    download_m3u8_file(first_m3u8_url, "越狱第一季第一集_first_m3u8.txt")
    # 3.2 下载第二层m3u8文件
    with open("越狱第一季第一集_first_m3u8.txt", mode="r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            else:
                line = line.strip()  # 去掉空白或者换行符  hls/index.m3u8
                # 准备拼接第二层m3u8的下载路径
                # https://boba.52kuyun.com/20170906/Moh2l9zV/ + hls/index.m3u8
                # https://boba.52kuyun.com/20170906/Moh2l9zV/hls/index.m3u8
                # https://boba.52kuyun.com/20170906/Moh2l9zV/hls/cFN8o3436000.ts
                second_m3u8_url = first_m3u8_url.split("index.m3u8")[0] + line
                download_m3u8_file(second_m3u8_url, "越狱第一季第一集_second_m3u8.txt")
                print("m3u8文件下载完毕")

    # 4. 下载视频
    second_m3u8_url_up = second_m3u8_url.replace("index.m3u8", "")
    # 异步协程
    asyncio.run(aio_download(second_m3u8_url_up))  # 测试的使用可以注释掉

    # 5.1 拿到秘钥
    key_url = second_m3u8_url_up + "key.key"  # 偷懒写法, 正常应该去m3u8文件里去找
    key = get_key(key_url)
    # 5.2 解密
    asyncio.run(aio_dec(key))

    # 6. 合并ts文件为mp4文件
    merge_ts()


if __name__ == '__main__':
    url = "https://www.91kanju.com/vod-play/541-2-1.html"
    main(url)
    # 简单的问题复杂化, 复杂的问题简单化
    # 秒杀()

