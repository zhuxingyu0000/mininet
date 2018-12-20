MODEL_DIR="model_dir/"
MODEL_META_NAME="MyModel.meta"

INPUT_ARRAY_DIR="c_sourcefile/"
INPUT_ARRAY_NAME="arr.txt"

import tensorflow as tf
import numpy as np

inputlist=[]

with open(INPUT_ARRAY_DIR+INPUT_ARRAY_NAME,'r',encoding='utf-8') as f:
	s=f.read()
	slist=s.split()
	inputlist=[float(i) for i in slist]

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(MODEL_DIR+MODEL_META_NAME)
	saver.restore(sess,tf.train.latest_checkpoint(MODEL_DIR))
	inputlist=np.array(inputlist).reshape(1,784)
	print(sess.run(tf.get_collection('out'),feed_dict={'Placeholder:0':inputlist,'Placeholder_2:0':1.0}))
