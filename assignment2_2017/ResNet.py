import tensorflow as tf 
import tensorflow.contrib.slim as slim 


def ResNet(input):
	"""
	This implementation is only for understanding the ResNet structure
	and basics to use tensorflow/slim to build neural network. 
	"""
	with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
		net = ResUnit(input, 64, [3, 3], 1, normalizer_fn=slim.batch_norm,
						scope='ResBlock1')
		for i in range(5):
			net = ResBlock(net, 'ResBlock' + str(i+2))

		net = bottleneck(net, 'bottleneck')


def ResBlock(input, scope):
	"""
	A ResBlock: conv2d + batch_norn + relu
	"""
	with tf.varaible_scope(scope):
		net = ResUnit(input, 64, [3, 3], 1, False, scope=(scope+"/unit1"))
		net = ResUnit(input, 64, [3, 3], 1, False, scope=(scope+"/unit2"))
		net = ResUnit(input, 64, [3, 3], 1, True, scope=(scope+"/unit3"))

		net += input
		net = tf.nn.relu(net)

		return net


def ResUnit(input, depth, kernel_size, stride, lastUnit, scope):
	"""
	A ResUnit: conv2d + batch_norn + relu(optional)
	"""
	with tf.variable_scope(scope):
		net = slim.conv2d(input, depth, kernel_size, stride=stride,
						padding='VALID',
						normalizer_fn=slim.batch_norm,
						scope=(scope+"/conv2d"))
		if not lastUnit:
			net = tf.nn.relu(net)

		return net


def bottleneck(input, scope):
	"""
	bottleneck: pool + fc + softmax
	"""
	net = slim.avg_pool(input, [2, 2], 2, scope=(scope+"pool"))
	net = slim.fc(net, 10, scope=(scope+"fc"))
	net = slim.layers.softmax(net)

	return net