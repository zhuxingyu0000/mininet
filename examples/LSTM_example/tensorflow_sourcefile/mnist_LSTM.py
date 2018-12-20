import tensorflow as tf
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("../../datasets/mnist/",one_hot=True)
sess=tf.InteractiveSession()

time_steps=28

x=tf.placeholder(tf.float32,[None,784])
x_in=tf.reshape(x,[-1,28,28])
y_=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

LSTMcell=tf.nn.rnn_cell.LSTMCell(num_units=100,forget_bias=1.0,state_is_tuple=True)
LSTMcell=tf.nn.rnn_cell.DropoutWrapper(cell=LSTMcell,input_keep_prob=1.0,output_keep_prob=keep_prob)

mlstm_cell=tf.nn.rnn_cell.MultiRNNCell([LSTMcell],state_is_tuple=True)

init_state=mlstm_cell.zero_state(1,dtype=tf.float32)

outputs,state=tf.nn.dynamic_rnn(
	mlstm_cell,
	inputs=x_in,
	initial_state=init_state,
	time_major=False
)
final_out=outputs[:,-1,:]
W_conv1=tf.Variable(tf.truncated_normal([100,10],stddev=0.1),dtype=tf.float32)
b_conv1=tf.Variable(tf.constant(0.1,shape=[10]),dtype=tf.float32)
y_conv=tf.nn.softmax(tf.matmul(final_out,W_conv1)+b_conv1)
tf.add_to_collection('out',y_conv)
tf.add_to_collection('W1',tf.trainable_variables()[0])
tf.add_to_collection('U1',tf.trainable_variables()[1])
tf.add_to_collection('W_conv1',tf.trainable_variables()[2])
tf.add_to_collection('b_conv1',tf.trainable_variables()[3])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.global_variables_initializer().run()
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

for i in range(100000):
	batch=mnist.train.next_batch(1)
	train_step.run({x:batch[0],y_:batch[1],keep_prob:0.5})
	print("训练进度:{0}%".format(round(i+1)*100.0/100000),end="\r")
"""
time_steps=1
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,10])
x_in=tf.reshape(x,[-1,1,10])
y_=tf.placeholder(tf.float32,[None,2])
LSTMcell=tf.nn.rnn_cell.LSTMCell(num_units=2,forget_bias=1.0,state_is_tuple=True)

mlstm_cell=tf.nn.rnn_cell.MultiRNNCell([LSTMcell],state_is_tuple=True)

init_state=mlstm_cell.zero_state(1,dtype=tf.float32)

outputs,state=tf.nn.dynamic_rnn(
	mlstm_cell,
	inputs=x_in,
	initial_state=init_state,
	time_major=False
)
final_out=outputs[:,-1,:]

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(final_out),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.global_variables_initializer().run()
correct_prediction=tf.equal(tf.argmax(final_out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

batch=[[[1,2,3,4,5,6,7,8,9,10]],[[8,3]]]

tf.add_to_collection('out',final_out)
tf.add_to_collection('W1',tf.trainable_variables()[0])
tf.add_to_collection('U1',tf.trainable_variables()[1])
tf.add_to_collection('aaa',outputs)

for i in range(10):
	train_step.run({x:batch[0],y_:batch[1]})
saver = tf.train.Saver()
saver.save(sess,"../model_dir/MyModel")

