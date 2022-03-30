import requests
from bs4 import BeautifulSoup

url = 'https://www.bqkan8.com/1_1496/'

resp = requests.get(url)
resp.encoding = 'gbk'

page = BeautifulSoup(resp.text, "html.parser")