import requests
from bs4 import BeautifulSoup
import time
url = "https://www.umeitu.com/bizhitupian/weimeibizhi/"

resp = requests.get(url)
resp.encoding = 'utf-8'
page = BeautifulSoup(resp.text, "html.parser")
table = page.find("div", class_="TypeList")
href_list = table.find_all("a")
for a in href_list:
    href = a.get("href")
    new_url = url+ href.strip("bizhitupian/weimeibizhi/")+ "htm"
    child_resp = requests.get(new_url)
    child_resp.encoding = 'utf-8'
    child_page = BeautifulSoup(child_resp.text, "html.parser")
    img_area = child_page.find("div", class_="ImageBody")
    img = img_area.find("img")
    imglink = img.get("src")
    imgresp = requests.get(imglink)
    imgresp.content
    img_name= imglink.split("/")[-1]
    with open("img/"+img_name, mode="wb") as f:
        f.write(imgresp.content)
    print("over!!","img_name")
    time.sleep(1)
    f.close()
print("done")

resp.close()