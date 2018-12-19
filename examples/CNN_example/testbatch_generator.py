INPUT_ARRAY_DIR="c_sourcefile/"
INPUT_ARRAY_NAME="arr.txt"
MNIST_DATA_DIR="tensorflow_sourcefile/MNIST_data/"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets(MNIST_DATA_DIR,one_hot=True)

with open(INPUT_ARRAY_DIR+INPUT_ARRAY_NAME,'w',encoding='utf-8') as f:
	data=mnist.test.images[1]
	for i in data:
		f.write(str(i)+' ')
	data=data.reshape(28,28)
	import matplotlib.pyplot as plt
	plt.imshow(data)
	plt.show()
