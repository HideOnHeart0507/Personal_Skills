import requests
import re

url = "https://www.dydytt.net/index2.htm"
headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"
}
resp = requests.get(url, headers=headers)

resp.encoding ='gb2312'
page_html = resp.text


obj = re.compile(r'2022新片精品.*?<ul>(?P<link>.*?)</ul>', re.S)
result = obj.finditer(page_html)
obj2 = re.compile(r"<a href='(?P<chaolianjie>.*?)'",re.S)
child_href_list=[]

for i in result:
    ui = i.group('link')
    # print(i.group('link'))`
    #拿到ul里的li
    result2 = obj2.finditer(ui)
    for j in result2:
        child_href= 'htt'+url.strip('index2.htm')+j.group("chaolianjie")
        child_href_list.append(child_href)

for href in child_href_list:
    child_resp = requests.get(href, headers=headers)
    child_resp.encoding='gb2312'
    obj3 = re.compile(r'◎片　　名(?P<pianming>.*?)<br />.*?<a target="_blank" href="(?P<lj>.*?)"><strong>', re.S)
    result3 =obj3.finditer(child_resp.text)
    for i in result3:
        print(i.group('pianming'))
        print(i.group("lj"))
resp.close()