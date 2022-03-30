import numpy as np
import operator
import collections

def create_data_set():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels =['love','love','action','action']
    return group,labels

def classify0(inx, dataset, labels, k):
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	group, labels = create_data_set()
	test = [101,20]
	test_class = classify0(test, group, labels, 3)
	print(test_class)