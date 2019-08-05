###################### Part I (Rule based) ######################
# --- Rule based (西部世界里面的机器人是怎么讲话的)
simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phrase => verb noun_phrase
Article => 一个 | 这个
noun => 女人 | 篮球 | 桌子 | 小猫
verb => 看着 | 坐在 | 听着 | 看见
Adj => 蓝色的 | 好看的 | 小小的
"""
simplest_grammar = """
number = number number | single_number
single_number = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0
"""
import random
def adj(): return random.choice('蓝色的 | 好看的 | 小小的'.split('|')).split()[0]
def adj_star(): ## 生成多个adj，*在正则表达式里面会讲，star就是指一个或者多个的意思。
	return random.choice([lambda : '', lambda : adj() + adj_star()])()
	return random.choice([None, adj() + adj()])()
adj_star()
## 问题是：当更换了语法，会发现所有的程序都要重新写
adj_grammar = """
Adj* => null | Adj Adj*
Adj => 蓝色的 | 好看的 | 小小的
"""
# terminal指的是不能扩展的东西，所有左边的东西代表可以继续扩展的，如果是右边就代表停止扩展了。
grammar = {}
for line in adj_grammar.split('\n'):
	if not line.strip(): continue
	exp,stmt = line.split('=>')
	grammar[exp] = stmt.split('|')
## 优化上面的程序
grammar = {}
for line in adj_grammar.split('\n'):
	if not line.strip(): continue
	exp,stmt = line.split('=>')
	grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
grammar
grammar['Adj*']
## 生成语法的函数
choice = random.choice
def generate(gram, target):
	if target in gram:
		newexpanded = random.choice(gram[target])
		return ' '.join(generate(gram, t) for t in newexpanded)
	else:
		return target
## 优化上面的程序
choice = random.choice
def generate(gram, target):
	if target not in gram: return target # means target is a terminal expression
	expaned = [generate(gram, t) for t in choice(gram[target])]
	return ' '.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])
generate(gram=grammar, target = 'Adj*')
## 将上面的过程抽象成一个函数
def create_grammar(grammar_str, split='=>', line_split='\n'):
	grammar = {}
	for line in grammar_str.split(line_split):
		if not line.strip(): continue
		exp, stmt = line.split(split)
		grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
	return grammar
example_grammar = create_grammar(simplest_grammar)
generate(gram=example_grammar, target = 'Adj*')

## 一个”人类“的语言可以定义为：
human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""
## 一个“接待员”的语言可以定义为：
host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""
for i in range(20):
	print(generate(gram=create_grammar(human, split='='), target='human'))
	print(generate(gram=create_grammar(host, split='='), target='host'))
# --- 我们的目标：
# 	Generalization: 希望做一个程序，当输入的数据变化的时候，我们的程序不用重新写。
# 	AI: 研究AI的人都在思考如何可以自动化解决问题，希望找到一个方法，输入变了以后，我们的这个方法不用变。
# --- 生成程序
programming = """
stmt => if_exp | while_exp | assignment 
assignment => var = var
if_exp => if ( var ) { /n .... stmt }
while_exp=> while ( var ) { /n .... stmt }
var => chars number
chars => char | char char
char => student | name | info | database | course
number => 1 | 2 | 3
"""
print(generate(gram=create_grammar(programming, split='=>'), target='stmt'))

###################### Part II (Probability based) ######################
# --- 以上基于语法规则我们已经可以生成完整的句子，那么如何生成最合理的一句话？
# --- Probability based
# 	根据语言定义可以生成句子，但是没办法把句子逆向判断是不是符合语法是不可能的，因为语法太复杂了。
# 	语言模型：language model。可以想象成一个函数。
# 		函数的输入是一个句子，输出是一个【0，1】之间的值，越接近于1代表越接近于人话。
# 		这个语言模型怎么做的呢？基于概率去做的。
# 		Automata：组合很复杂
# --- 基于概率的方法介绍：
# language_model(String) = Probability(String) 范围为 (0, 1)
# Pro(w1w2w3w4) = Pr(w1|w2w3w4) * P(w2|w3w4) * Pr(w3|w4) * Pr(w4)
# Pro(w1w2w3w4) ～ Pr(w1|w2) * P(w2|w3) * Pr(w3|w4) * Pr(w4)
# how to get Pr(w1|w2w3w4)？
# count(w1)/count(w2w3w4) # w1出现在w2w3w4之后
# --- 判断一句话会不会出现，只需要判断这个词组会不会出现，然后再计算出来这个概率就可以了。
# --- sqlResult_1558435.csv这个文件是爬虫得到的新闻事件的文件
# jupyter只是一个实验环境，当文件太大的时候就会死机。如果开发大型项目，还是要在pycharm里面写。
filename = './sqlResult_1558435.csv'
import pandas as pd
content = pd.read_csv(filename, encoding='gb18030') ## 具体的编码应该是一个一个试出来的，没有别的很好的办法。
content.head()
articles = content['content'].tolist()
len(articles)
articles[110]
## 接下来需要做的就是：根据这些话去判断这些单词出现的概率。在此之前需要去除非法字符，也就是无用字符。需要的工具就是正则表达式，下一节课会介绍正则表达式，正则表达式就是定义一种模式，这种模式会把所有满足这个模式的东西找出来。w+就是找出来所有的单词。
import re
def token(string):
	# we will learn the regular expression next course.
	return re.findall('\w+', string)
''.join(token(articles[110])) ## 取出无意字符之后连接起来文字。
##
articles_clean = [''.join(token(str(a)))for a in articles] # 有些是空值，所以要加str()
len(articles_clean)
## 直接使用结巴分词过程 ##
## 直接使用结巴分词会有一些问题，出现最多的就是无意义的那些东西。
from collections import Counter
import jieba # 结巴分词的原理很复杂，准确率已经很高，所以很少专门研究结巴分词了。
with_jieba_cut = Counter(jieba.cut(articles[110]))
with_jieba_cut.most_common()[:10]
##
with open('article_9k.txt', 'w') as f:
	for a in articles_clean:
		f.write(a + '\n')
def cut(string): return list(jieba.cut(string))
## 下面的过程对电脑配置要求高 ##
article_words= [
	cut(string) for string in articles_clean
]
article_words
article_words[0]
## 统计每个单词出现的频数
from functools import reduce
from operator import add
reduce(add, [1, 2, 3, 4, 5, 8])
[1, 2, 3] + [3, 43, 5]
TOKEN = []
## 运行代码时，卡在了下面一步，因为运行的时间太久。超过两个小时，有可能哪里出错了。！！！
TOKEN = reduce(add, article_words) ## 就可以知道所有单词的出现频率了。
##
for i, line in enumerate((open('article_9k.txt'))):
	if i % 100 == 0: print(i)
	# replace 10000 with a big number when you do your homework. 
	if i > 10000: break
	TOKEN += cut(line)
len(TOKEN)


## 以上为数据预处理
from collections import Counter
words_count = Counter(TOKEN)
words_count.most_common(100) ## 出现频率最大的100个字
frequiences = [f for w, f in words_count.most_common(100)]
## 可视化数据
x = [i for i in range(100)]
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(x, frequiences) ## 指数曲线，统计学现象，极少数的单词占据了出现最多的频率。
import numpy as np
plt.plot(x, np.log(frequiences)) ## 取了对数
## 求某一个单词出现的频率。
def prob_1(word):
	return words_count[word] / len(TOKEN)
prob_1('我们')
## 
TOKEN[:10]
TOKEN = [str(t) for t in TOKEN]
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
TOKEN_2_GRAM[:10]
words_count_2 = Counter(TOKEN_2_GRAM)
## 定义一个函数，可以求两个单词一起出现的频率。
def prob_2(word1, word2):
	if word1 + word2 in words_count_2: 
		return words_count_2[word1+word2] / len(TOKEN_2_GRAM)
	else:
		# return (prob_1(word1) + prob_1(word2)) / 2 ## 正常应该这么定义
		return 1 / len(TOKEN_2_GRAM) ## 但是因为我们数据量的限制，先怎么定义
prob_2('我们', '在')
prob_2('在', '吃饭')
prob_2('去', '吃饭')
## 抽象出一个函数，计算一个句子出现的概率 ##
def get_probablity(sentence):
	words = cut(sentence)
	sentence_pro = 1
	for i, word in enumerate(words[:-1]):
		next_ = words[i+1]	
		probability = prob_2(word, next_)
		sentence_pro *= probability
	return sentence_pro
get_probablity('小明今天抽奖抽到一台苹果手机')
get_probablity('小明今天抽奖抽到一架波音飞机')
get_probablity('洋葱奶昔来一杯')
get_probablity('养乐多绿来一杯')
## 计算根据某一个语法生成的句子出现的概率大小
for sen in [generate(gram=example_grammar, target='sentence') for i in range(10)]:
	print('sentence: {} with Prb: {}'.format(sen, get_probablity(sen)))
##
need_compared = [
	"今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
	"真事一只好看的小猫 真是一只好看的小猫",
	"今晚我去吃火锅 今晚火锅去吃我",
	"洋葱奶昔来一杯 养乐多绿来一杯"
]
for s in need_compared:
	s1, s2 = s.split()
	p1, p2 = get_probablity(s1), get_probablity(s2)
	better = s1 if p1 > p2 else s2
	print('{} is more possible'.format(better))
	print('-'*4 + ' {} with probility {}'.format(s1, p1))
	print('-'*4 + ' {} with probility {}'.format(s2, p2))
  
 
######################## Part III 基础理论部分: #########################
# 	0. Can you come up out 3 sceneraies which use AI methods?
# 	Ans: {儿童玩具火火兔，火车站的人脸识别自动验票，苹果手机的Siri}

# 	1. How do we use Github; Why do we use Jupyter and Pycharm;
# 	Ans: {GitHub用于共享开源项目，代码等。Jupyter是一个很好的演示工具，不适合在上面跑大型数据；Pycharm可以用于大型项目的开发。}

# 	2. Whats the Probability Model?
# 	Ans: {标准定义：对于语言序列w1,w2,w3,...,wn,语言模型就是计算该序列的概率，即P(w1,w2,...,wn)。从机器学习的角度来看：语言模型是对语句的概率分布建模。通俗来讲就是判断一个语言序列是否符合我们人类正常语句表达。我的理解：概率模型可以根据已有的语言总结出来每一个单词出现的频率以及两个单词相邻出现的概率，从而就可以基于一段单词连续出现的概率去生成句子。}

# 	3. Can you came up with some sceneraies at which we could use Probability Model?
# 	Ans: {例如统计人类基因组中某些碱基序列（例如CG序列）出现的概率}

# 	4. Why do we use probability and whats the difficult points for programming based on parsing and pattern match?
# 	Ans: {我的理解是要考虑的情况很多。}

# 	5. Whats the Language Model;
# 	Ans: {简单来说语言模型就是一串词序列的概率分布。在实践中，如果文本的长度较长，P(wi|w1w2,...,wi-1)的估算会很困难，因此研究者们提出来了简化模型，即n元模型（n-gram model），在n元模型中估算条件概率时，只需要对当前词的前n个词进行计算。在n元模型中，传统的方法一般采用频率计数的比例来估算n元条件概率。当n较大时，即会存在数据稀疏的问题，导致估算结果不准确。因此一般在百万词级别的语料中，一般也就用到三元模型}

# 	6. Can you came up with some sceneraies at which we could use Language Model?
# 	Ans: {我们在键盘上打字的时候，当打出来一个字，输入法上可能会根据字与字之间连接的概率以及某个字词出现的频率从而可以补全等。例如，当我们在键盘上按了两次"d",就会出来“对了”，“等等“等这些词组。}

# 	7. Whats the 1-gram language model;
# 	Ans: {n-gram model模型基于一种假设，即第N个词的出现只与前面N-1个词相关，而与其他任何词都不相关，整个句子的概率就是各个词出现概率的乘积。这些概率可以从语料中统计N个词同时出现的次数得到。1-gram模型就是n = 1时的模型。对于1-gram，其假设是P(wn|w1w2…wn-1)≈P(wn|wn-1)。在1-gram模型下：P(w1, w2, w3, … , wn) = P(w1)P(w2|w1)P(w3|w1w2)P(w4|w1w2w3)…P(wn|w1w2…wn-1) ≈ P(w1)P(w2|w1)P(w3|w2)P(w4|w3)…P(wn|wn-1)。假设我们采用的是1-gram模型，那么：P(I,have,a,gun)=P(I)P(have|I)P(a|have)P(gun|a). 然后，我们再用“数数”的方法求P(I)和其他的三个条件概率：P(I)=语料库中I出现的次数 / 语料库中的总词数。P(have|I) = 语料库中I和have一起出现的次数 / 语料库中I出现的次数。}

# 	8. Whats the disadvantages and advantages of 1-gram language model;
# 	Ans: {优点：包含了前面1个词所能提供的全部信息，这些词对于当前词的出现有很强的约束力。缺点就是需要相当规模的训练样本来确定模型的参数。}

# 	9. Whatt the 2-gram models;
# 	Ans: {对于2-gram，其假设是P(wn|w1w2…wn-1)≈P(wn|wn-1,wn-2)。在2-gram模型下：P(w1, w2, w3, … , wn) = P(w1)P(w2|w1)P(w3|w1w2)P(w4|w1w2w3)…P(wn|w1w2…wn-1) ≈ P(w1)P(w2|w1)P(w3|w1w2)P(w4|w2w3)…P(wn|wn-2wn-1)}

######################## Part IV 编程实践部分 #########################
# 1. 设计你自己的句子生成器
# 如何生成句子是一个很经典的问题，从1940s开始，图灵提出机器智能的时候，就使用的是人类能不能流畅和计算机进行对话。和计算机对话的一个前提是，计算机能够生成语言。
# 计算机如何能生成语言是一个经典但是又很复杂的问题。我们课程上为大家介绍的是一种基于规则（Rule Based）的生成方法。该方法虽然提出的时间早，但是现在依然在很多地方能够大显身手。值得说明的是，现在很多很实用的算法，都是很久之前提出的，例如，二分查找提出与1940s, Dijstra算法提出于1960s 等等。
# 在著名的电视剧，电影《西部世界》中，这些机器人们语言生成的方法就是使用的SyntaxTree生成语言的方法。
# ##
host = """
host = 主人公 时间 位置 做事情 
时间 = 早晨 | 中午 | 晚上 | 下午
位置 = 在公园里 | 在郊野外 | 在山上 | 在草原上 ， 
主人公 = 我 | 你 | 很多人 | 老人们 | 孩子们
做事情 = 动作
动作 = 运动 | 跑步 | 呼吸新鲜空气 | 游览 | 休息 | 打闹
"""
test = """
test = 感慨 , 描述 展望
感慨 = 时光飞逝 | 光阴似箭 | 日月如梭 | 时间过得好快啊 | 日子一天天过去
描述 = 主语 就要 动作 ,
主语 = 我们 | 你们 | 他们 | 孩子们 | 学生们 | 老师们 | 这学期
动作 = 毕业 | 远走他乡 | 长大 | 出国 | 考试
展望 = 希望 主语 好词
好词 = 越来越好 | 越来越幸福 | 前程似锦 | 万里鹏程 !
"""
##
import random
choice = random.choice
##
def create_grammar(grammar_str, split='=>', line_split='\n'):
	grammar = {}
	for line in grammar_str.split(line_split):
		if not line.strip(): continue
		exp, stmt = line.split(split)
		grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
	return grammar
##
def generate(gram, target):
	if target not in gram: return target # means target is a terminal expression
	expaned = [generate(gram, t) for t in choice(gram[target])]
	return ''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])
##
def generate_n(n, gram, target):
	i = 0
	sentences = []
	while i < int(n):
		if target not in gram: return target # means target is a terminal expression
		expaned = [generate(gram, t) for t in choice(gram[target])]
		sentences += [''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])]
		i += 1
	return sentences
##
for i in range(20):
	print(generate(gram=create_grammar(host, split='='), target='host'))
	print(generate(gram=create_grammar(test, split='='), target='test'))
##
print(generate_n(20, gram=create_grammar(host, split='='), target='host'))
print(generate_n(20, gram=create_grammar(test, split='='), target='test'))


# 2. 使用新数据源完成语言模型的训练
# 按照我们上文中定义的prob_2函数，我们更换一个文本数据源，获得新的Language Model:
# 下载文本数据集（你可以在以下数据集中任选一个，也可以两个都使用）
# 可选数据集1，保险行业问询对话集： https://github.com/Computing-Intelligence/insuranceqa-corpus-zh/raw/release/corpus/pool/train.txt.gz
# 可选数据集2：豆瓣评论数据集：https://github.com/Computing-Intelligence/datasource/raw/master/movie_comments.csv
# 修改代码，获得新的2-gram语言模型
# 进行文本清洗，获得所有的纯文本
# 将这些文本进行切词
# 送入之前定义的语言模型中，判断文本的合理程度
# 实际上只需要将Part II中的代码的传入文件进行一下替换即可。

import pandas as pd
import re
filename = '/data/tusers/lixiangr/caill/NLP/data/sqlResult_1558435.csv'
content = pd.read_csv(filename, encoding='gb18030')
content.head()
##
def token(string):
	# we will learn the regular expression next course.
	return re.findall('\w+', string)
''.join(token(articles[110])) ## 取出无意字符之后连接起来文字。
## 获得纯文本 ##
articles_clean = [''.join(token(str(a)))for a in articles] # 有些是空值，所以要加str()
len(articles_clean)
##
with open('article_9k.txt', 'w') as f:
	for a in articles_clean:
		f.write(a + '\n')
def cut(string): return list(jieba.cut(string))
## 分词 ##
article_words= [
	cut(string) for string in articles_clean
]

## 以下为统计每个单词出现的频数 ##
from functools import reduce
from operator import add
TOKEN = []
##
for i, line in enumerate((open('article_9k.txt'))):
	if i % 100 == 0: print(i)
	# replace 10000 with a big number when you do your homework. 
	if i > 100000: break
	TOKEN += cut(line)
len(TOKEN)

## 2-gram语言模型 ##
from collections import Counter
words_count = Counter(TOKEN)
words_count.most_common(100) ## 出现频率最大的100个字
frequiences = [f for w, f in words_count.most_common(100)]
## 可视化数据
x = [i for i in range(100)]
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(x, frequiences) ## 指数曲线，统计学现象，极少数的单词占据了出现最多的频率。
import numpy as np
plt.plot(x, np.log(frequiences)) ## 取了对数
## 求某一个单词出现的频率。
def prob_1(word):
	return words_count[word] / len(TOKEN)
prob_1('我们')
## 
TOKEN[:10]
TOKEN = [str(t) for t in TOKEN]
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
TOKEN_2_GRAM[:10]
words_count_2 = Counter(TOKEN_2_GRAM)
## 定义一个函数，可以求两个单词一起出现的频率。
def prob_2(word1, word2):
	if word1 + word2 in words_count_2: 
		return words_count_2[word1+word2] / len(TOKEN_2_GRAM)
	else:
		# return (prob_1(word1) + prob_1(word2)) / 2 ## 正常应该这么定义
		return 1 / len(TOKEN_2_GRAM) ## 但是因为我们数据量的限制，先怎么定义
prob_2('我们', '在')
prob_2('在', '吃饭')
prob_2('去', '吃饭')
## 抽象出一个函数，基于2-gram语言模型去计算一个句子出现的概率 ##
def get_probablity(sentence):
	words = cut(sentence)
	sentence_pro = 1
	for i, word in enumerate(words[:-1]):
		next_ = words[i+1]	
		probability = prob_2(word, next_)
		sentence_pro *= probability
	return sentence_pro
get_probablity('小明今天抽奖抽到一台苹果手机')
get_probablity('小明今天抽奖抽到一架波音飞机')
get_probablity('洋葱奶昔来一杯')
get_probablity('养乐多绿来一杯')
## 计算根据某一个语法生成的句子出现的概率大小
for sen in [generate(gram=example_grammar, target='sentence') for i in range(10)]:
	print('sentence: {} with Prb: {}'.format(sen, get_probablity(sen)))
##
need_compared = [
	"今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
	"真事一只好看的小猫 真是一只好看的小猫",
	"今晚我去吃火锅 今晚火锅去吃我",
	"洋葱奶昔来一杯 养乐多绿来一杯"
]
for s in need_compared:
	s1, s2 = s.split()
	p1, p2 = get_probablity(s1), get_probablity(s2)
	better = s1 if p1 > p2 else s2
	print('{} is more possible'.format(better))
	print('-'*4 + ' {} with probility {}'.format(s1, p1))
	print('-'*4 + ' {} with probility {}'.format(s2, p2))

# 3. 获得最优质的的语言
# 当我们能够生成随机的语言并且能判断之后，我们就可以生成更加合理的语言了。请定义 generate_best 函数，该函数输入一个语法 + 语言模型，能够生成n个句子，并能选择一个最合理的句子:
# 提示，要实现这个函数，你需要Python的sorted函数

sentences = generate_n(20, gram=create_grammar(host, split='='), target='host')
def generate_best(sentences): # you code here
	sentences_prob = []
	for i in range(len(sentences)):
		# sentences_prob.append((get_probablity(sentences[i][0]), sentences[i][0]))
		sentences_prob.append((get_probablity(sentences[i][0])))
	# return sorted(sentences_prob, key = lambda x: x[0], reverse=True)[0]
	max_index = sentences_prob.index(max(sentences_prob))
	return sentences[max_index],sentences_prob

generate_best(sentences)










