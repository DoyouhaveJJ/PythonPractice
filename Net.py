import tensorflow as tf
import math 
import numpy as np
import matplotlib.pyplot as plt
import types
import pylab

def draw_correct_line():
	#画sin曲线
	x = np.arange(0,2*np.pi,0.01)
	x = x.reshape((len(x),1))
	y = np.sin(x)

	pylab.plot(x,y,label = '标准的sin曲线')
	plt.axhline(linewidth=1,color='r')

def get_train_data():
	#返回训练样本 train_x trian_y
	#trian_x随机的自变量 train_y为trian_y的sin函数值
	train_x = np.random.uniform(0.0,2*np.pi,(1))#uniform解释：从0.0到2*pi随机(1)个取样 uniform(low,high,size)其中左闭右开
	train_y = np.sin(train_x)
	return train_x,train_y

def inference(input_data):
	#定义网络结构
	#Args:输入的input_data值 单
	with tf.variable_scope('hidden1'):
		#第一个隐藏层
		#get_variable(name名字，shape形状,dtype数据类型,initializer,regularizer,trainable(是否可训练),collections,
		#		caching_device,partitioner,…………)
		#tf.random_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
		#			平均值，标准偏差的随机生成 ，创建随机，数据类型
		#返回一个具有正态分布的张量的初始化器
		weights  = tf.get_variable('weight',[1,16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		biases = tf.get_variable('bias',[1,16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		#隐藏层1 神经网络的sigmoid y=1/(1+exp(-x))  weight*input_data+bias
		hidden1 = tf.sigmoid(tf.multiply(input_data,weights) + biases)

	
	with tf.variable_scope('hidden2'):
		#隐藏层2
		weights  = tf.get_variable('weight',[16,16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		biases = tf.get_variable('bias',[16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		#矩阵乘法
		mul = tf.matmul(hidden1,weights)
		hidden2 = tf.sigmoid(mul + biases)

	with tf.variable_scope('hidden3'):
		#隐藏层3
		weights  = tf.get_variable('weight',[16,16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		biases = tf.get_variable('bias',[16],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		hidden3 = tf.sigmoid(tf.matmul(hidden2,weights) + biases)

	with tf.variable_scope('output_layer'):
		#输出层
		weights  = tf.get_variable('weight',[16,1],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		biases = tf.get_variable('bias',[1],tf.float32,
			initializer = tf.random_normal_initializer(0.0,1))
		output = tf.matmul(hidden3,weights) + biases
	return output

def train():
	#学习率
	learning_rate = 0.01
	#x为输入值
	x = tf.placeholder(tf.float32)
	#y为训练
	y = tf.placeholder(tf.float32)
	
	net_out = inference(x)
	
	#损失函数
	loss_op = tf.square(net_out - y)
	#优化器，为随机梯度下降法
	opt = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = opt.minimize(loss_op)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		print("训练中…………")
		for i in range(10000):
			train_x,train_y = get_train_data()
			sess.run(train_op,feed_dict={x:train_x,y:train_y})
			
			if(i%100==0):
				times = int(i/100)
				test_x_ndarray = np.arange(0,2*np.pi,0.01)
				test_y_ndarray = np.zeros([len(test_x_ndarray)])
				ind = 0
		for test_x in test_x_ndarray:
			test_y = sess.run(net_out,feed_dict={x:test_x,y:1})
			np.put(test_y_ndarray,ind,test_y)
			ind+=1
		draw_correct_line()
		pylab.plot(test_x_ndarray,test_y_ndarray,'--',label = str(times)+'times')
		pylab.show()
if __name__ == "__main__":
	train()
		
