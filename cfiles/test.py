import tensorflow as tf

input = tf.constant(
[
        [
                [
                        [100., 100., 100.],
                        [100., 100., 100.],
                        [100., 100., 100.]
                ],
                [
                        [100., 100., 100.],
                        [100., 100., 100.],
                        [100., 100., 100.]
                ],
                [
                        [100., 100., 100.],
                        [100., 100., 100.],
                        [100., 100., 100.],
                ]
        ]
]
);


filter = tf.constant(
[
	[
		[[0.5],
		[0.5],
		[0.5]]
	]       
]
);

print(input.shape,filter.shape)

result = tf.nn.convolution(input, filter, padding='SAME')

with tf.Session() as sess:
		print(sess.run(result))
		x=sess.run(result)
		print(x.shape)