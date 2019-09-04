## 190904 Assignment8


### Recode (交叉熵)

```python

import numpy as np
from collections import Counter

def entropy(elements):
	counter = Counter(elements)
	probs = [counter[c] / len(elements) for c in elements]
	return -sum(p * np.log(p) for p in probs)

entropy([0, 0, 1])

```


### 数据预处理

这里的数据预处理主要包括去除文件中的缺失值。原本新闻语料中含有89611个新闻，但是其中只有87052个新闻内容('content')不为空。另外，还需要给数据添加标签。新华社的新闻标记为1，其余来源的新闻标记为0。

这里可以读取已经分好词的文件，节省时间。也可以处理的时候再一句句分词。

```python
import pandas as pd
csv_path = "/data/tusers/lixiangr/caill/NLP/data/sqlResult_1558435.csv"
news = pd.read_csv(csv_path, encoding = 'gb18030')
news_nona = news.dropna(subset = ['source', 'content'])

# 标签
source = news_nona['source'].tolist()
y = [1 if source[i] == '新华社' else 0 for i in range(len(news_nona))]
y_ = pd.Series([y], index=['y'])

# 读取已经分好词的文件，并且插入y一列
corpus = pd.read_csv("/data/tusers/lixiangr/caill/NLP/data/news-sentences-cut.txt", header = None, sep = "\t")
corpus.columns = ['content']
corpus.insert(0, 'y', y)
news_content = corpus['content'].tolist()

```

```python

y.count(0) # 8391
y.count(1) # 78661

# pos/neg = 9.374448814205696

```

### 构建新闻文本的TF-IDF向量

这里需要使用所有的新闻预料构建TF-IDF向量。然后再去划分验证集、测试集和训练集。生成的X即TF-IDF向量，X的每一行代表一条新闻，每一列代表每一个词。可以使用max_features控制所使用的词的数目。对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集。

```python

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 500)
X = vectorizer.fit_transform(news_content)

X.shape
X.toarray()
word = vectorizer.get_feature_names()
```

划分验证集（validation）、测试集（test）和训练集（training）。后面我们使用training数据集进行训练模型，validation数据进行验证。

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1)

X_train.shape
# (62676, 500)
X_val.shape
# (15670, 500)
X_test.shape
# (8706, 500)
```

### 支持向量机 SVM

关于支持向量机，在[这里](https://www.studyai.cn/modules/svm.html)有非常详细的描述。在[这里](http://www.stardustsky.net/index.php/post/53.html)有比较详细的参数介绍。

* 支持向量机的优点:

在高维空间中非常高效。
即使在数据维度比样本数量大的情况下仍然有效。
在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的.
通用性: 不同的核函数与特定的决策函数一一对应.常见的 kernel 已经提供,也可以指定定制的内核.

* 支持向量机的缺点:

如果特征数量比样本数量大得多, 选择核函数 核函数 和 正则化项以避免过拟合是很关键的。
支持向量机不直接提供概率估计,这些都是使用昂贵的五次交叉验算计算的。 (详情见 Scores and probabilities, 在下文中)

SVC, NuSVC 和 LinearSVC 能够在指定的数据集上进行多类分类任务的类。
SVC 和 NuSVC 实现了 “one-against-one” 方法 (Knerr et al., 1990) 用于解决多类别分类问题。

* 得分与概率

对于大数据集，Platt scaling 方法中使用交叉验证是一个昂贵操作。而且，使用SVM的得分进行概率估计得到的结果是不一致的(inconsistent), 从这个意义上说，得分的最大化并不等价于概率的最大化(the “argmax” of the scores may not be the argmax of the probabilities)。 (比如说, 在二分类问题中, 一个样本可能会被 predict 标记为属于其中一个根据 predict_proba 估计出的概率 < 1/2的类(a sample may be labeled by predict as belonging to a class that has probability < 1/2 according to predict_proba.)。 Platt的方法还被认为存在一些理论问题。 如果我们需要信任得分(confidence scores), 但是这些信任得分不一定是概率性得分，那么建议设置 probability=False 并使用 decision_function 而不是 predict_proba。

* decision_function：基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示测试样本到对应类型的超平面距离。

* 参数gamma ： 它是一个浮点数，作为三个核函数的参数，隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，。gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。‘rbf’,‘poly’和‘sigmoid’的核函数参数。默认是’auto’，如果是auto，则值为1/n_features。

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train, y_train)  #SVM的训练过程很慢

pred = clf.predict(X_val)
pred_prob = clf.decision_function(X_val) # 这里使用得分

# score
clf.score(X_val, y_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9012763241863433

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 1.0

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9480750511865204

# ROC-AUC
roc_auc_score(y_val, pred_prob)
# 0.9763331952751797

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob)
area = auc(recall, precision)
# 0.9973629650196316
```

* 另外，SVM很大的一个优点就是可以解决数据的非平衡问题。

在某些问题中，我们需要给某些类或个别样本更大的重要性，这时候就要使用关键参数 class_weight 和 sample_weight 。SVC (but not NuSVC) 在 fit 方法中实现了关键字参数 class_weight，是一个形式为 {class_label : value} 的字典, 其中 value 是一个大于0的浮点数，把 class_label 对应的类的参数 C 设置为 C * value。

* 参数class_weight：字典类型或者‘balance’字符串,默认为None。 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面指出的参数C.

当class_weight = "balanced"时，对于每个样本，计算的损失函数乘上对应的sample_weight来计算最终的损失。这样计算而来的损失函数不会因为样本不平衡而被“推向”样本量偏少的类别中。

* 参数C：这个值官方文档说的比较明白了，就是在泛化能力和准确度间做取舍，一般来说不需要做修改，如果不需要强大的泛化能力，可减小C的值，即：C值越大，在测试集效果越好，但可能过拟合，C值越小，容忍错误的能力越强，在测试集上的效果越差。参数C和gamma是svm中两个非常重要的参数，对其值的调整直接决定了整个模型最终的好坏。

C的默认值是1.0，C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

因为我们数据中正类和负类的比例约为10:1。在 SVC 类中, 如果用于分类的数据是不均衡的(unbalanced) (比如 很多的正样本但是负样本很少), 请设置 class_weight='balanced' and/or 尝试不同的惩罚参数(正则化) C 。


```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

from sklearn.svm import SVC

clf = SVC(gamma='auto', class_weight = "balanced")

clf.fit(X_train, y_train)  #SVM的训练过程很慢

pred = clf.predict(X_val)
pred_prob = clf.decision_function(X_val) # 这里使用得分

# score
clf.score(X_val, y_val)
# 0.7580089342693044

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9993233446109232

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.7319974509665085

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.8450220696419813

# ROC-AUC
roc_auc_score(y_val, pred_prob)
# 0.9601502745227417

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob)
auc(recall, precision)
# 0.9955361578079207
```

从上面的结果可以看出，预测效果整体来说下降了。因为准确率下降；精准率很高但是召回率降低。F1 score只有0.84。ROC-AUC提升。


* 参数kernel，我们这里并没有改变kernel，而是 使用默认的高斯（RBF）核函数：不同的核函数对最后的分类效果影响也比较大，其中precomputed表示自己提前计算好核函数矩阵，这时候算法内部就不再用核函数去计算核矩阵，而是直接用你给的核矩阵。

* 如何选用不同的高斯核函数？

因此，在选用核函数的时候，如果我们对我们的数据有一定的先验知识，就利用先验来选择符合数据分布的核函数；如果不知道的话，通常使用交叉验证的方法，来试用不同的核函数，误差最下的即为效果最好的核函数，或者也可以将多个核函数结合起来，形成混合核函数。在吴恩达的课上，也曾经给出过一系列的选择核函数的方法：

如果特征的数量大到和样本数量差不多，则选用LR或者线性核的SVM；
如果特征的数量小，样本的数量正常，则选用SVM+高斯核函数；
如果特征的数量小，而样本的数量很大，则需要手工添加一些特征从而变成第一种情况。


### 随机森林 RF

随机森林的参数详解在[这里](https://www.cnblogs.com/pinard/p/6160412.html)。

这里可以和GBDT对比来学习。在scikit-learn [梯度提升树(GBDT)调参小结](https://www.cnblogs.com/pinard/p/6143927.html)中我们对GBDT的框架参数做了介绍。GBDT的框架参数比较多，重要的有最大迭代器个数，步长和子采样比例，调参起来比较费力。但是RF则比较简单，这是因为bagging框架里的各个弱学习器之间是没有依赖关系的，这减小的调参的难度。换句话说，达到同样的调参效果，RF调参时间要比GBDT少一些。


RF主要的框架参数:
* 参数n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值。默认是10。

* oob_score :
即是否采用袋外样本来评估模型的好坏。默认识False。个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。

* criterion: 
即CART树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。一般来说选择默认的标准就已经很好的。

从上面可以看出， RF重要的框架参数比较少，主要需要关注的是 n_estimators，即RF最大的决策树个数。

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train) 

pred = clf.predict(X_val)
pred_prob = clf.predict_proba(X_val)

# score
clf.score(X_val, y_val)
# 0.9822590938098277

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9895339792093911

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9901641664307954

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9480750511865204

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.9901741468813954

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:, 1])
auc(recall, precision)
# 0.9988552374950626
```
我感觉不调节参数，效果已经很好了！

但是还是试试去调节参数：oob_score=True, class_weight='balanced'。模型效果基本不变。### Recode (交叉熵)



```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(oob_score=True, class_weight='balanced')

clf.fit(X_train, y_train) 
pred = clf.predict(X_val)
pred_prob = clf.predict_proba(X_val)

# score
clf.score(X_val, y_val)
# 0.9839183152520741

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9915656673045574

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9905827373787439

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9910739586285067

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.987537898290488

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:, 1])
auc(recall, precision)
# 0.9985635348441135
```




















































