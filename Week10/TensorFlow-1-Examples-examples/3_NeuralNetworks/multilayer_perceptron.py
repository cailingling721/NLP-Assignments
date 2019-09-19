'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
"""
# tf.nn.softmax_cross_entropy_with_logits:首先pred经过softmax函数变为类别概率，然后再计算交叉熵，相当于softmax和交叉熵的计算合二为一。
# 预测越准确， （- 求和（真实y * log（预测y ））），即交叉熵的结果值越小（别忘了前面还有负号）
# 这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作, 就是对向量里面所有元素求和。
# 如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！

# 逻辑回归
# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
# 交叉熵的计算：- 求和（真实y * log(预测y)）
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
"""

"""
sigmoid 和 softmax的对比：refer to https://blog.nex3z.com/2017/05/02/sigmoid-函数和-softmax-函数的区别和关系/
    Sigmoid 函数形式为：
        𝑆(𝑥)=1/（1+𝑒−𝑥）
        Sigmoid 是一个可微的有界函数，在各点均有非负的导数。当 𝑥→∞ 时，𝑆(𝑥)→1；当 𝑥→−∞ 时，𝑆(𝑥)→0。
       常用于二元分类（Binary Classification）问题，以及神经网络的激活函数（Activation Function）（把线性的输入转换为非线性的输出）。  
    Softmax 函数形式为：
        𝑆(𝑥𝑗)=𝑒𝑥𝑗∑𝐾𝑘=1𝑒𝑥𝑘,𝑗=1,2,…,𝐾
        对于一个长度为 K 的任意实数矢量，Softmax 可以把它压缩为一个长度为 K 的、取值在 (0, 1) 区间的实数矢量，且矢量中各元素之和为 1。
        它在多元分类（Multiclass Classification）和神经网络中也有很多应用。Softmax 不同于普通的 max 函数：max 函数只输出最大的那个值，
        而 Softmax 则确保较小的值也有较小的概率，不会被直接舍弃掉，是一个比较“Soft”的“max”。
"""
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
