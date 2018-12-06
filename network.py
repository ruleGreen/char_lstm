# -*- coding:UTF-8 -*-

"""
神经网络模型
RNN_LSTM 循环神经网络
"""

import tensorflow as tf

# 神经网络的模型
class Model(object):
	# 构造函数
	def __init__(self,input,is_training,hidden_size,vocab_size,num_layers,dropout=0.5,init_scale=0.05):
		self.is_training = is_training
		self.input_obj = input
		self.batch_size = input.batch_size
		self.num_steps = input.num_steps
		self.hidden_size = hidden_size
		
		# 让这里的操作和变量用CPU来计算,因为暂时还没有GPU的实现
		with tf.device("/cpu:0"):
			# 创建词向量(Word Embedding),Embedding表示Dense Vector(密集向量)
			embedding = tf.Variable(tf.random_uniform([vocab_size,self.hidden_size],-init_scale,init_scale))
			# embedding_lookup返回词向量
			inputs = tf.nn.embedding_lookup(embedding,self.input_obj.input_data)

		# 如果是　训练时　并且dropout率小于1,使输入经过一个dropout层
		if is_training and dropout<1:
			inputs = tf.nn.dropout(inputs,dropout)

		# 状态(state)的存储与提取
		# 第二维是2是因为对每一个LSTM单元有两个来自上一单元的输入:
		# 一个是　前一时刻LSTM的输出h(t-1)
		# 一个是　前一时刻的单元状态C(t-1)
		# 这个　C和h 是用于构建之后的 tf.contrib.rnn.LSTMStateTuple
		self.init_state = tf.placeholder(tf.float32,[num_layers,2,self.batch_size,self.hidden_size])

		# 每一层的状态
		state_per_layer_list = tf.unstack(self.init_state,axis=0)

		# 初始的状态(包含前一时刻的两个输入，用于之后的dynamic_rnn)
		rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1]) for idx in range(num_layers)])

		# 创建一个　LSTM层,其中的神经元数目是 hidden_size个（默认是650个）
		cell = tf.contrib.rnn.LSTMCell(hidden_size)

		# 如果是训练时并且Dropout率小于1,给LSTM层加上Dropout操作
		# 这里只给　输出　加了Dropout操作,留存率(output_keep_prob)是0.5
		# 输入是默认的1,所以相当于输入没有做Dropout操作
		if is_training and dropout<1 :
			cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=dropout)

		# 如果LSTM的层数大于1,则总计创建num_layers个LSTM层
		# 并将所有的LSTM层包装进MultiRNNCell这样的序列化层级模型中
		# state_is_tuple=True表示接受　LSTMStateTuple形式的输入状态
		if num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)],state_is_tuple=True)

		# dynamic_rnn(动态RNN)可以让不同迭代传入的Batch可以是长度不同的数据
		# 但同一次迭代中一个Batch内部的所有数据长度仍然是固定的
		# dynamic_rnn能更好的处理padding(补0)的情况,节约计算资源
		# 返回两个变量
		# 第一个是一个　Batch　里在时间维度(默认是35)上展开的所有LSTM单元的输出，形状默认为[20,35,650],之后会经过扁平层处理
		# 第二个是最终的state(状态)，包含当前时刻的LSTM的输出h(t) 和当前时刻的单元状态C(t)
		output,self.state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32,initial_state=rnn_tuple_state)

		# 扁平化处理　改变输出形状为 (batch_size*num_steps,hidden_size),形状默认为[700,650]
		output = tf.reshape(output,[-1,hidden_size])

		# Softmax 的权重(Weight)
		softmax_w = tf.Variable(tf.random_uniform([hidden_size,vocab_size],-init_scale,init_scale))

		# Softmax 的偏置(Bias)
		softmax_b = tf.Variable(tf.random_uniform([vocab_size],-init_scale,init_scale))

		# logits　是Logistic Regression(用于分类)模型(线性方程:y=w*x+b)计算的结果
		# 这个logits(分值)之后会用Softmex来转为百分比概率
		# output是输入(x)
		# 返回w*x+b的结果
		logits = tf.nn.xw_plus_b(output,softmax_w,softmax_b)

		# 将logits转化为三维的Tensor，为了sequence loss的计算
		# 形状默认为[20,35,10000]
		logits = tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])

		# 计算logits的交叉熵(Cross-entropy)的损失
		loss = tf.contrib.seq2seq.sequence_loss(
				logits,   # 形状默认为[20,35,10000]
				self.input_obj.targets, # 期望输出 [20,35]
				tf.ones([self.batch_size,self.num_steps],dtype=tf.float32),
				average_across_timesteps = False,
				average_across_batch=True)

		# 更新代价
		self.cost = tf.reduce_sum(loss)

		# Softmax算出来的概率
		self.softmax_out = tf.nn.softmax(tf.reshape(logits,[-1,vocab_size])) # logits形状[700,10000]

		# 取最大概率的那个值作为预测
		self.predict = tf.cast(tf.argmax(self.softmax_out,axis=1),tf.int32)

		# 预测值与真实值(目标)对比
		correct_prediction = tf.equal(self.predict,tf.reshape(self.input_obj.targets,[-1]))

		# 计算预测的精度
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

		# 如果是　测试,则直接退出
		if not is_training:
			return

		# 学习率, trainable=False 表示不可被训练
		self.learning_rate = tf.Variable(0.0,trainable=False)
	
		# 返回所有可被训练(trainable=True.如果不设置trainable=False,默认的Variable都是可以被训练的)
		# 也就是除了不可以被训练的学习率之外的所有变量
		tvars = tf.trainable_variables()

		# tf.clip_by_global_norm(实现Gradient Clipping(梯度裁剪))是为了防止梯度爆炸
		# tf.gradients 计算self.cost对于tvars的梯度(求导),返回一个梯度的列表
		grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),5)

		# 优化器用 GradientDescentOptimizer(梯度下降优化器)
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

		# apply_gradients(应用梯度)将之前(Gradient Clipping)梯度裁剪过得梯度应用到可被训练的变量,做梯度下降
		# apply_gradients其实是minimize方法里面的第二步,第一步是计算梯度
		self.train_op = optimizer.apply_gradients(
				zip(grads,tvars),
				global_step=tf.train.get_or_create_global_step())

		# 用于更新学习率
		self.new_lr = tf.placeholder(tf.float32,shape=[])
		self.lr_update = tf.assign(self.learning_rate,self.new_lr)

	# 更新学习率
	def assign_lr(self,session,lr_value):
		session.run(self.lr_update,feed_dict={self.new_lr:lr_value})
