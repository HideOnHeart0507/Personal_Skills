from lxml import etree

tree = etree.parse("a.html")
result = tree.xpath("html/body/h3/text()")
print(result)