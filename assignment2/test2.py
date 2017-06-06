__author__ = 'T7'

__author__ = 'T7'

import numpy as np
import matplotlib.pyplot as plt
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.im2col import *

""" test conv_forwar
x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

x_col = im2col.im2col_indices(x, w.shape[2], w.shape[3], p, s)
"""

x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)

p = 1
s = 1

num_filters, _, filter_height, filter_width = w.shape

dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)

print dout_reshaped.shape



