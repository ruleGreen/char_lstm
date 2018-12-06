# -*- coding:UTF-8 -*-

from utils import *
from network import *

def train(train_data,vocab_size,num_layers,num_epochs,batch_size,model_save_name,learning_rate=1.0,max_lr_epoch=10,lr_decay=0.93,print_iter=50):
	# 训练的输入
	training_input = Input(batch_size=batch_size,num_steps=35,data=train_data)

	# 创建训练的模型
	m = Model(training_input,is_training=True,hidden_size=650,vocab_size=vocab_size,num_layers=num_layers)

	# 初始化变量的操作
	init_op = tf.global_variables_initializer()

	# 初始的学习率(learning rate)的衰减率
	orig_decay = lr_decay

	with tf.Session() as sess:
		sess.run(init_op)

		# Coordinator(协调器) 用于协调线程的运行
		coord = tf.train.Coordinator()
		# 启动线程
		threads = tf.train.start_queue_runners(coord=coord)

		# 为了用Saver来保存模型的变量
		saver = tf.train.Saver() # max_to_keep默认是5，只保存最近的5个模型参数文件
		
		# 开始epoch的训练
		for epoch in range(num_epochs):
			new_lr_decay = orig_decay ** max(epoch+1-max_lr_epoch,0.0) #前max_lr_epoch的学习率为1,之后衰减
			m.assign_lr(sess,learning_rate*new_lr_decay)
			current_state = np.zeros((num_layers,2,batch_size,m.hidden_size)) #第二维2代表LSTM cell的两个输入
			# 获取当前时间　打印日志
			curr_time = datetime.datetime.now()

			for step in range(training_input.epoch_size):
				# train_op操作:计算被修剪后的梯度,并最小化误差
				# state操作:返回时间维度上展开的最后LSTM单元的输出(C(t)和H(t)),作为下一个Batch的输入状态
				if step % print_iter != 0 :
					cost,_,current_state = sess.run([m.cost,m.train_op,m.state],feed_dict={m.init_state:current_state})
				else:
					seconds = (float((datetime.datetime.now() - curr_time).seconds) / print_iter)
					curr_time = datetime.datetime.now()
					cost,_,current_state,acc = sess.run([m.cost,m.train_op,m.state,m.accuracy],feed_dict={m.init_state:current_state})
					# 每print_iter打印当下的cost和accuracy
					print("Epoch {}, Step {}, Cost:{:.3f},Accuracy:{:.3f},Seconds per step: {:.3f}".format(epoch,step,cost,acc,seconds))
			# 保存一个模型的变量的checkpoint文件
			saver.save(sess,save_path+'/'+model_save_name,global_step=epoch)

		# 对模型做一次总的保存
		saver.save(sess,save_path+'/'+model_save_name+'-final')

		# close thread
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	if args.data_path:
		data_path = args.data_path

	train_data,valid_data,test_data,vocab_size,id_to_word = load_data(data_path)
	train(train_data,vocab_size,num_layers=2,num_epochs=10,batch_size=20,model_save_name='train-checkpoint')
