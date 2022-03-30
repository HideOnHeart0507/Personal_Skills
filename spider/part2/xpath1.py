#xpath 搜索xml文档内容 html是xml子集
# 安装lxml

from lxml import etree

xml = """ <book>
    <id>1</id>
</book>
"""

tree = etree.XML(xml)
# root = tree.xpath("/book/id")
# root = tree.xpath("/book/id/text()")取文本
# root1 = tree.xpath("/book/author//nick/text()") 取后面全部得子、后代
root = tree.xpath("/book/id")
# root1 = tree.xpath("/book/author/*/nick/text()") 通配符 任意节点
# root1 = tree.xpath("/book/author/li[1]/nick/text()") 中括号中数字检索索引 1是开始
# root1 = tree.xpath("/html/body/ol/li/a[@href='dapao']") @后接标签内的属性=指定内容
print(root)

ollilist = tree.xpath("html/body/ol/li")
# for i in ollilist:
#     print(i)
#     i.xpath("./a/text") 在li中继续去寻找
#     i.xpath("./a/@href") 拿属性的值