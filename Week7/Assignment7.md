## 190903 Assignment7

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


### KNN



在使用不同的模型拟合数据后，我们可以进行预测。此时预测有三种方法，包括predict，predict_log_proba和predict_proba。

* predict方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。

* predict_proba则不同，它会给出测试集样本在各个类别上预测的概率。容易理解，predict_proba预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。

* predict_log_proba和predict_proba类似，它会给出测试集样本在各个类别上预测的概率的一个对数转化。转化后predict_log_proba预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别。

建模之后，使用准确率、查准率、查全率、F1 score、PR-AUC以及ROC-AUC进行评估。


```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

neigh = KNeighborsClassifier(n_neighbors=3) # n_neighbors的缺省值是5
neigh.fit(X_train, y_train) 

neigh.score(X_val, y_val)
pred = neigh.predict(X_val)
pred_prob = neigh.predict_proba(X_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9390169284111768

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.977979182893153

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9581021087680355

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.8018761064085546

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
area = auc(recall, precision)
# 0.9778175881204517
```

#### KNN调参


* K值的选择与样本分布有关，一般选择一个较小的K值，可以通过交叉验证来选择一个比较优的K值，默认值是5。如果数据是三维一下的，如果数据是三维或者三维以下的，可以通过可视化观察来调参。

* 参数近邻权weights：主要用于标识每个样本的近邻样本的权重，如果是KNN，就是K个近邻样本的权重，如果是限定半径最近邻，就是在距离在半径以内的近邻样本的权重。可以选择"uniform","distance" 或者自定义权重。选择默认的"uniform"，意味着所有最近邻样本权重都一样，在做预测时一视同仁。如果是"distance"，则权重和距离成反比例，即距离预测目标更近的近邻具有更高的权重，这样在预测类别或者做回归时，更近的近邻所占的影响因子会更加大。当然，我们也可以自定义权重，即自定义一个函数，输入是距离值，输出是权重值。这样我们可以自己控制不同的距离所对应的权重。

一般来说，如果样本的分布是比较成簇的，即各类样本都在相对分开的簇中时，我们用默认的"uniform"就可以了，如果样本的分布比较乱，规律不好寻找，选择"distance"是一个比较好的选择。如果用"distance"发现预测的效果的还是不好，可以考虑自定义距离权重来调优这个参数。

* KNN和限定半径最近邻法使用的算法algorithm ：算法一共有三种，第一种是蛮力实现，第二种是KD树实现，第三种是球树实现。对于这个参数，一共有4种可选输入，‘brute’对应第一种蛮力实现，‘kd_tree’对应第二种KD树实现，‘ball_tree’对应第三种的球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法。需要注意的是，如果输入样本特征是稀疏的时候，无论我们选择哪种算法，最后scikit-learn都会去用蛮力实现‘brute’。

个人的经验，如果样本少特征也少，使用默认的 ‘auto’就够了。 如果数据量很大或者特征也很多，用"auto"建树时间会很长，效率不高，建议选择KD树实现‘kd_tree’，此时如果发现‘kd_tree’速度比较慢或者已经知道样本分布不是很均匀时，可以尝试用‘ball_tree’。而如果输入样本是稀疏的，无论你选择哪个算法最后实际运行的都是‘brute’。

* 停止建子树的叶子节点阈值leaf_size：这个值控制了使用KD树或者球树时， 停止建子树的叶子节点数量的阈值。这个值越小，则生成的KD树或者球树就越大，层数越深，建树时间越长，反之，则生成的KD树或者球树会小，层数较浅，建树时间较短。默认是30. 这个值一般依赖于样本的数量，随着样本数量的增加，这个值必须要增加，否则不光建树预测的时间长，还容易过拟合。可以通过交叉验证来选择一个适中的值。

* 距离度量metric ：K近邻法和限定半径最近邻法类可以使用的距离度量较多，一般来说默认的欧式距离（即p=2的闵可夫斯基距离）就可以满足我们的需求。

从上述结果中我们可以看出只有ROC-AUC比较低。按理来说应该先去分析为什么PR-AUC比较高 ，但是ROC-AUC比较低。所以下面去尝试新的参数。将n_neighbors调整为5。


```python
neigh = KNeighborsClassifier(n_neighbors=5) # n_neighbors的缺省值是5
neigh.fit(X_train, y_train) 

neigh.score(X_val, y_val)
pred = neigh.predict(X_val)
pred_prob = neigh.predict_proba(X_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9410354745925216

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9729519223960915

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9567275892080069

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.8864503802381525

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
area = auc(recall, precision)
# 0.9864611444916045
```

从上面的结果我们可以看出：ROC-AUC有很大提升，同时其余的评估指标也有提升，总体来说效果已经很好。我就认为调整参数的方向是对的。下面再次尝试将 n_neighbors 调整为7，看看效果。

```python
neigh = KNeighborsClassifier(n_neighbors=7) # n_neighbors的缺省值是5
neigh.fit(X_train, y_train) 

neigh.score(X_val, y_val)
pred = neigh.predict(X_val)
pred_prob = neigh.predict_proba(X_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9372578342736727

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9762798272321744

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9563709509606714

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.9000309452263087

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
area = auc(recall, precision)
# 0.9880128918540668
```

从上面的结果可以看出，ROC-AUC仍旧有一些提升。此外，我还尝试了将n_neighbors调整为9，基本上各个评估参数已经保持不变。

另外，对于KNN来说还有另外一个重要参数，即近邻权重 weights。默认的"uniform"，意味着所有最近邻样本权重都一样，在做预测时一视同仁。如果是"distance"，则权重和距离成反比例，即距离预测目标更近的近邻具有更高的权重，这样在预测类别或者做回归时，更近的近邻所占的影响因子会更加大。这里我尝试将weights调整为distance。

```python
neigh = KNeighborsClassifier(n_neighbors=7, weights = 'distance') # n_neighbors的缺省值是5
neigh.fit(X_train, y_train) 

neigh.score(X_val, y_val)
pred = neigh.predict(X_val)
pred_prob = neigh.predict_proba(X_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9421898406645786

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9797493450400057

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9606025894685688

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.9179012756198074

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
area = auc(recall, precision)
# 0.9901032113225808
```

从上述结果可以看出，参数又有小幅度的提升。


### 朴素贝叶斯

[朴素贝叶斯方法](https://www.cnblogs.com/pinard/p/6074222.html)是一系列有监督学习的方法，这些方法基于对贝叶斯理论的应用，即简单(naive)的假设 每对特征之间都相互独立。

通过[scikit-learn官网API说明](http://sklearn.lzjqsdd.com/modules/naive_bayes.html)可以看到，sklearn将贝叶斯的三个常用模型都封装好了，分别是：高斯贝叶斯（Gaussian Naive Bayes）、多项式贝叶斯（Multinomial Naive Bayes）、伯努利贝叶斯（Bernoulli Naive Bayes）。接着就可以学习它所给出的例程了。

各种各样的朴素贝叶斯分类器的不同之处在于，他们对 P(x_i|y) 的分布的认识和假设不同。

尽管它们看起来有一个过于简单的假设，朴素贝叶斯分类器仍然 在真实世界的许多情景下工作良好，在文本分类和垃圾邮件筛选领域尤其流行。 它们要求少量的数据来估计必要的参数。 (关于理论上朴素贝叶斯为什么会工作良好，以及它可以适用的数据类型，详见下方References)

朴素贝叶斯学习和分类器与其他相比可以非常快。类条件特征分布的解耦意味着 每个分布可以独立估计为一个一维分布，这反过来又有助于缓解维灾难问题。

另一方面，虽然被称为一个合适的分类器，它也被认为是是一个坏的估计量，所以对 predict_proba 的概率输出不应太过依赖。

#### 朴素贝叶斯 高斯模型

下面尝试使用高斯贝叶斯建模。高斯贝叶斯一般适用于样本分布符合或者类似高斯分布的时候使用。

高斯模型假设这些一个特征的所有属于某个类别的观测值符合高斯分布。GaussianNB类的主要参数仅有一个，即先验概率priors ，对应Y的各个类别的先验概率𝑃(𝑌=𝐶𝑘)。这个值默认不给出，如果不给出此时𝑃(𝑌=𝐶𝑘)=𝑚𝑘/𝑚。其中m为训练集样本总数量，𝑚𝑘为输出为第k类别的训练集样本数。如果给出的话就以priors 为准。


```python

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train.toarray(), y_train)
pred = clf.predict(X_val.toarray())
pred_prob = clf.predict_proba(X_val.toarray())

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9906658369632856

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.789067478581038

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.8784486835882075

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.8962313785693254

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
auc(recall, precision)
# 0.98782882967254
```

此外，GaussianNB一个重要的功能是有 partial_fit方法，这个方法的一般用在如果训练集数据量非常大，一次不能全部载入内存的时候。这时我们可以把训练集分成若干等分，重复调用partial_fit来一步步的学习训练集，非常方便。后面讲到的MultinomialNB和BernoulliNB也有类似的功能。

#### 朴素贝叶斯 多项式模型

MultinomialNB 实现了数据服从多项式分布时的贝叶斯算法，它也是文本分类领域的 两种典型算法之一(这里数据通常以词向量的形式表示，tf-idf向量在这里也表现的很好)。 这个分布被参数化成向量：

```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train.toarray(), y_train)
pred = clf.predict(X_val.toarray())
pred_prob = clf.predict_proba(X_val.toarray())

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9532094469534316

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9116335056291156

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9319580166485704

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.9102314273603492

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
auc(recall, precision)
# 0.9896928463464106
```

#### 伯努利模型

类 BernoulliNB 实现了对于服从多元伯努利分布的数据的朴素贝叶斯训练和分类算法； 也就是说，对于大量特征，每一个特征都是一个0-1变量 (Bernoulli, boolean)。 因此，这个类要求样本集合以0-1特征向量的方式展现。如果接收到了其他类型的数据作为参数， 一个 BernoulliNB 实例会把输入数据二元化(取决于 binarize 参数设置)

在文本分类的情境中，被用来训练和使用这一分类器的是词语同现向量 (word occurrence vectors) 而不是词频向量 (word count vectors)。 BernoulliNB 可能尤其会在小数据集时表现良好。


```python
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()

clf.fit(X_train.toarray(), y_train)
pred = clf.predict(X_val.toarray())
pred_prob = clf.predict_proba(X_val.toarray())

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9674234945705824

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.8326842738794874

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.8950112256935195

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.8858861253203398

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
auc(recall, precision)
# 0.9869821857107934
```

通过试用以上所有朴素贝叶斯模型，我可以看出的是多项式模型在此套新闻语料文本分类上效果更好。


### 逻辑回归

```python

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

clf.fit(X_train, y_train) 

pred = clf.predict(X_val)
pred_prob = clf.predict_proba(X_val)

# precision = TP/(TP + FP)
precision_score(y_val, pred) 
# 0.9776720095191432

# recall = TP/(TP + FN)
recall_score(y_val, pred)
# 0.9890249946895135

# 1/F1 = 1/2 * (1/P + 1/R) or 1/F1 = 2PR/(P + R)
f1_score(y_val, pred)
# 0.9833157338965153

# ROC-AUC
roc_auc_score(y_val, pred_prob[:, 1])
# 0.9891842749550868

# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:,1])
area = auc(recall, precision)
# 0.9987933183957046
```

使用逻辑回归对此文本分类的效果最佳。

### 寻找抄袭新华社的文章

我们可以首先使用逻辑回归的模型去判断一篇文章是否是新华社的文章，如果判断出来是新华社的，但是，它的source并不是新华社的，那么，我们就说，这个文章是抄袭的新华社的文章。即predict的label为1，但是实际上label为0的新闻。

```python

pred_all = clf.predict(X)

plagiarized = []
for i in range(len(y)):
	if pred_all[i] == 1 and y[i] == 0:
		plagiarized.append(i)

news_nona.index = range(len(y))
plagiarized_news = news_nona.ix[plagiarized, :]
len(plagiarized) # 1643
```

因为样本的正类和负类比例大约为10:1，所以倾向于有更多的负类会归为正类，也就是误判率（假阳性率）会比较高。所以应适当提升逻辑回归中设定的正类阈值。

