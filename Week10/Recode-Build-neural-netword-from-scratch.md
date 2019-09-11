## Recode - Build neural network from scratch

定义一个基类：

```py
class Node:
	"""
	Each node in neural networks will have these attributes and methods
	"""
	def __init__(self, inputs=[]):
		"""
		if the node is the operator of "ax + b", the inputs will be x node , and the outputs 
		of this is its successors. 
		
		and the value is *ax + b*
		"""
		self.inputs = inputs # input_list <- C, Java <- 匈牙利命名法 -> Python 特别不建议。# 匈牙利命名法,变量名字里面写进去变量类型。但是python里面不建议，在C里面直接就是错的。因为通不过编译。python里的话不好读。
	   # self.outputs = outputs # output_list 
		self.value = None # 当前自己的值是多少？ax + b
		self.outputs = []
		self.gradients = {}
		
		for node in self.inputs:
			node.outputs.append(self) # build a connection relationship
	
	def forward(self):
		"""Forward propogation
		
		compute the output value based on input nodes and store the value 
		into *self.value*
		"""
		raise NotImplemented # 相当于是一个虚类，没有实现，如果一个对象是它的子类，那么就必须重现实现这个方法。
	
	def backward(self):
		""" Back propogation
		
		compute the gradient of each input node and store the value 
		into "self.gredients"
		"""
		raise NotImplemented

```

x --> wx+b --> sigmoid
input -> linear -> activation
所以还有三个节点，即input、linear和activation。

Node的一个子类：

```py
class Input(Node):
	def __init__(self, name=''):
		Node.__init__(self, inputs=[])
		self.name = name
	
	def forward(self, value=None):
		if value is not None:
			self.value = value
		
	def backward(self):
		self.gradients = {}
		
		for n in self.outputs:
			grad_cost = n.gradients[self] # n.gradients[self]是n对于self的偏导
			self.gradients[self] = grad_cost
	
	def __repr__(self):
		return 'Input Node: {}'.format(self.name)
```

```py
class Linear(Node):
	def __init__(self, nodes, weights, bias):
		self.w_node = weights
		self.x_node = nodes
		self.b_node = bias
		Node.__init__(self, inputs=[nodes, weights, bias])
	
	def forward(self): 
		"""compute the wx + b using numpy"""
		self.value = np.dot(self.x_node.value, self.w_node.value) + self.b_node.value
		
	
	def backward(self):
		
		for node in self.outputs:
			# gradient_of_loss_of_this_output_node = node.gradient[self]
			grad_cost = node.gradients[self]
			
			self.gradients[self.w_node] = np.dot(self.x_node.value.T, grad_cost)
			self.gradients[self.b_node] = np.sum(grad_cost * 1, axis=0, keepdims=False)
			self.gradients[self.x_node] = np.dot(grad_cost, self.w_node.value.T)
```

```py
class Sigmoid(Node):
	def __init__(self, node):
		Node.__init__(self, [node])
		self.x_node = node
	
	def _sigmoid(self, x):
		return 1. / (1 + np.exp(-1 * x))
	
	def forward(self):
		self.value = self._sigmoid(self.x_node.value)
	
	def backward(self):
		y = self.value
		
		self.partial = y * (1 - y)
		
		for n in self.outputs:
			grad_cost = n.gradients[self]
			
			self.gradients[self.x_node] = grad_cost * self.partial
```

#### loss节点

```py
class MSE(Node):
	def __init__(self, y_true, y_hat):
		self.y_true_node = y_true
		self.y_hat_node = y_hat
		Node.__init__(self, inputs=[y_true, y_hat])
	
	def forward(self):
		y_true_flatten = self.y_true_node.value.reshape(-1, 1)
		y_hat_flatten = self.y_hat_node.value.reshape(-1, 1)
		
		self.diff = y_true_flatten - y_hat_flatten
		
		self.value = np.mean(self.diff**2)
		
	def backward(self):
		n = self.y_hat_node.value.shape[0]
		
		self.gradients[self.y_true_node] = (2 / n) * self.diff
		self.gradients[self.y_hat_node] =  (-2 / n) * self.diff
```

#### 输入一个图，先每一个点都前向和后向一遍。

怎么样前向和后向一遍呢？所以graph 是经过拓扑排序之后的 一个list。

```py
def training_one_batch(topological_sorted_graph):
	# graph 是经过拓扑排序之后的 一个list
	for node in topological_sorted_graph:
		node.forward()
		
	for node in topological_sorted_graph[::-1]:
		node.backward()
```

#### 拓扑排序的代码

```py

def topological_sort(data_with_value):
	feed_dict = data_with_value 
	input_nodes = [n for n in feed_dict.keys()]

	G = {}
	nodes = [n for n in input_nodes]
	while len(nodes) > 0:
		n = nodes.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}
		for m in n.outputs:
			if m not in G:
				G[m] = {'in': set(), 'out': set()}
			G[n]['out'].add(m)
			G[m]['in'].add(n)
			nodes.append(m)

	L = []
	S = set(input_nodes)
	while len(S) > 0:
		n = S.pop()

		if isinstance(n, Input):
			n.value = feed_dict[n]
			## if n is Input Node, set n'value as 
			## feed_dict[n]
			## else, n's value is caculate as its
			## inbounds

		L.append(n)
		for m in n.outputs:
			G[n]['out'].remove(m)
			G[m]['in'].remove(n)
			# if no other incoming edges add to S
			if len(G[m]['in']) == 0:
				S.add(m)
	return L

```

#### SGD优化器：

```py
def sgd_update(trainable_nodes, learning_rate=1e-2):
	for t in trainable_nodes:
		t.value += -1 * learning_rate * t.gradients[t]
```

#### 波士顿房价预测数据

```py

from sklearn.datasets import load_boston
data = load_boston()
X_ = data['data']
y_ = data['target']
X_[0]
y_[0]
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
n_features = X_.shape[1]
n_hidden = 10
n_hidden_2 = 10
X.shape

W1_, b1_ = np.random.randn(n_features, n_hidden), np.zeros(n_hidden)
W2_, b2_ = np.random.randn(n_hidden, 1), np.zeros(1)

```

#### 建立图连接：

```py

# Build Nodes in this graph
X, y = Input(name='X'), Input(name='y')  # tensorflow -> placeholder
W1, b1 = Input(name='W1'), Input(name='b1')
W2, b2 = Input(name='W2'), Input(name='b2')
#W3, b3 = Input(name='W3'), Input(name='b3')


# build connection relationship
linear_output = Linear(X, W1, b1)
sigmoid_output = Sigmoid(linear_output)
yhat = Linear(sigmoid_output, W2, b2)
loss = MSE(y, yhat)

```

#### 可以将图连接网络变得复杂：

```py

input_node_with_value = {  # -> feed_dict 
	X: X_, 
	y: y_, 
	W1: W1_, 
	W2: W2_, 
	b1: b1_, 
	b2: b2_
}
graph = topological_sort(input_node_with_value)
graph

```


```py
from sklearn.utils import resample
np.random.choice(range(100), size=10, replace=True)

def run(dictionary):
	return topological_sort(dictionary)
```


```py
losses = []
epochs = 5000
batch_size = 64
steps_per_epoch = X_.shape[0] // batch_size

for i in range(epochs):
	loss = 0
	
	for batch in range(steps_per_epoch):
		#indices = np.random.choice(range(X_.shape[0]), size=10, replace=True)
		#X_batch = X_[indices]
		#y_batch = y_[indices]
		X_batch, y_batch = resample(X_, y_, n_samples=batch_size)
		
		X.value = X_batch
		y.value = y_batch
		
#		 input_node_with_value = {  # -> feed_dict 
#			 X: X_batch, 
#			 y: y_batch, 
#			 W1: W1.value, 
#			 W2: W2.value, 
#			 b1: b1.value, 
#			 b2: b2.value,
#		 }
		
#		 graph = topological_sort(input_node_with_value)
		
		training_one_batch(graph)
		
		learning_rate = 1e-3
		
		sgd_update(trainable_nodes=[W1, W2, b1, b2], learning_rate=learning_rate)
		
		loss += graph[-1].value
		
	if i % 100 == 0:
		print('Epoch: {}, loss = {:.3f}'.format(i+1, loss/steps_per_epoch))
		losses.append(loss)

```

linear_output = Linear(X, W1, b1) 
sigmoid_output = Sigmoid(linear_output) 
yhat = Linear(sigmoid_output, W2, b2) 
loss = MSE(y, yhat)

```py

import matplotlib.pyplot as plt
plt.plot(losses)
W1.value
W2.value

X_[1]
x1 = Input()
x1.value = X_[1]

y_of_x1 =  Linear(Sigmoid(Linear(x1, W1, b1)), W2, b2)

W1.value.shape

def _sigmoid(x):
		return 1. / (1 + np.exp(-1 * x))

np.dot(_sigmoid(np.dot(X_[1], W1.value) + b1.value), W2.value) + b2.value

y_[1]

y_of_x1
```