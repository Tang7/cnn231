__author__ = 'T7'
import numpy as np
import random
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized


'''
W = np.array([[2.56,-2.13,1.78],[-1.34,2.76,2.91],[-1.11,2.35,1.62]])
X = np.array([[2,1,2,-1],[1,-1,2,1],[1,2,1,-2]])
y = np.array([[2],[2],[0],[2]])
'''

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 490



# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y= y_train[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))

X = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T


W = np.random.randn(10, 3073) * 0.0001


num_train = X.shape[1]
# print num_train
# minus correct score not label value !!!
delta = 1.0
diff = W.dot(X)
# print diff
correct_score = np.zeros(y.shape)
y_index = np.zeros(diff.shape)
for i in range(len(y)):
	correct_score[i] = diff[y[i], i]
	y_index[y[i],i] = 1.0

# print correct_score

correct_score -= delta

diff -= correct_score.T

# print diff

diff[diff <= 0] = 0
diff -= y_index * delta   # set the correctly classified index value to zero

# print dif

loss = np.sum(diff)/ num_train
loss += 0.5*np.sum(W*W)


# diff = np.clip(diff, 0.0, 1.0)  # should be set the >0 value to 1 !

diff[diff>0] = 1

# print diff

num_sum = np.sum(diff, axis=0)

# print num_sum

# print num_sum * y_index

# dW = diff.dot(X.T)

correct = num_sum * y_index

print correct[:,:2]

diff -= correct

# print diff

dW = diff.dot(X.T)/float(num_train)

# print dW

loss1, grad = svm_loss_naive(W, X, y, 1)

# print grad


loss2, grad2 = svm_loss_vectorized(W, X, y, 1)

# print grad2
print loss
print loss1
print loss2

differ = np.linalg.norm(grad - grad2, ord='fro')
print 'difference: %f' % differ

differ1 = np.linalg.norm(dW - grad, ord='fro')
print 'difference: %f' % differ1

differ2 = np.linalg.norm(dW - grad2, ord='fro')
print 'difference: %f' % differ2

a = range(0,9)

print a

p = np.random.shuffle(a)

print a[:5]

'''
sum1 = 0.5 * np.sum(W * W)

print sum1

# y = np.array([[2],[1],[0],[2]])

loss1, _ = svm_loss_naive(W, X, y,1)

print loss1-sum1

loss2, _ = svm_loss_vectorized(W, X, y, 1)

print loss2-sum1
'''

'''
d = c.clip(min= 0)

print d

e = np.sum(d, axis = 0)

print e
'''

