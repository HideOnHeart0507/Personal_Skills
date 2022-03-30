from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pydotplus
# from sklearn.externals.six import StringIO
from io import StringIO
def create_dict():
    with open('lenses.txt','r') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lenses_targets= []
    for i in lenses:
        lenses_targets.append(i[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:  # 提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)  # 打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)
    return lenses_pd,lenses_targets

# 让特征标签序列化
def dict_encoding(dict):
    le = LabelEncoder()
    for cols in dict.columns:
        dict[cols] = le.fit_transform(dict[cols])
    print(dict)
    return dict

# 绘制决策树
def tree_making():
    with open('lenses.txt','r') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lenses_targets= []
    for i in lenses:
        lenses_targets.append(i[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:  # 提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)  # 打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)
    le = LabelEncoder()
    for cols in lenses_pd.columns:
        lenses_pd[cols] = le.fit_transform(lenses_pd[cols])

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_targets)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names= clf.classes_, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf('treegraph.pdf')
    return clf


if __name__ == '__main__':
    
    # dict,targets = create_dict()
    # dict = dict_encoding(dict)
    clf = tree_making()
    print(clf.predict([[1,1,1,0]]))