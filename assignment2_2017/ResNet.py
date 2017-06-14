import tensorflow as tf 
import tensorflow.contrib.slim as slim 

# Problem related to using is_training and reuse in tensorflow
# https://github.com/tensorflow/tensorflow/issues/5663

def ResNet(input, is_training=False):
    net = ResUnit(input, 64, [3, 3], 1, False, is_training, scope='ResBlock1')
    for i in range(5):
        net = ResBlock(net, is_training, 'ResBlock' + str(i+2))
    
    net = bottleneck(net, 'bottleneck')
    
    return net

def ResBlock(input, is_training=False, scope=None):
    """
    ResBlock: A list of ResUnits plus the input
    """
    with tf.variable_scope(scope):
        net = ResUnit(input, 64, [3, 3], 1, False, is_training, scope=(scope+"/unit1"))
        net = ResUnit(input, 64, [3, 3], 1, False, is_training, scope=(scope+"/unit2"))
        net = ResUnit(input, 64, [3, 3], 1, True, is_training, scope=(scope+"/unit3"))
        
        net += input
        net = tf.nn.relu(net)
        
    return net

def ResUnit(input, depth, kernel_size, stride, lastUnit, is_training=False, scope=None):
    """
    ResUnit: conv2d + batch_norn + relu(optional)
    """
    with tf.variable_scope(scope):
        if lastUnit:
            net = slim.conv2d(input, depth, kernel_size, stride=stride,
                             padding='SAME',
                             activation_fn=tf.nn.relu,
                             normalizer_fn=slim.batch_norm,
                             normalizer_params={'is_training': is_training},
                             scope=(scope+"/conv2d"))
        else:
            net = slim.conv2d(input, depth, kernel_size, stride=stride,
                             padding='SAME',
                             activation_fn=None,
                             normalizer_fn=slim.batch_norm,
                             normalizer_params={'is_training': is_training},
                             scope=(scope+"/conv2d"))
            
    return net


def bottleneck(input, scope=None):
    net = slim.avg_pool2d(input, [2, 2], 2, scope=(scope+"/pool"))
    reshape_size = int(net.shape[1] * net.shape[2] * net.shape[3])
    net = tf.reshape(net,[-1,reshape_size])
    net = slim.fully_connected(net, 10, scope=(scope+"/fc"))
    net = slim.layers.softmax(net)
    
    return net