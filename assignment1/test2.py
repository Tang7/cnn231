__author__ = 'T7'

import numpy as np

a = np.array([3,1,2,0])

b = np.random.randint(5, size=(4,4))

print a

print b

c = np.arange(len(a))

print c

d = b[c,a]

print d

e = np.zeros(b.shape)

e[c, a] = 1

print e


