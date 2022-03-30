import re

#findall: 匹配字符串中所有符合的正则内容
lst = re.findall(r"\d+", "我的电话是:10086")
print(lst)

#finditer: 匹配字符串中所有的内容but迭代器

lst = re.finditer(r"\d+", "我的电话是:10086")
print(lst)
for i in lst:
    print(i)
    print(i.group())

#search返回match对象 只能拿第一个
s = re.search(r"\d+", "我的电话是:10086")
print(s)
print(s.group())

#match从头开始匹配
m = re.match(r"\d+", "我的电话是:10086")
print(m)

#预加载正则
# (P<分组名字>正则) 可以从正则表达式中进一步提取内容
obj = re.compile(r"(?P<number>\d+)", re.S) #让.能匹配换行fu
ob  =obj.finditer("我的电话是10086")
for i in ob:
    print(i.group("number"))