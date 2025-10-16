import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
import matplotlib.pyplot as plt
from pylab import cm
size = 784
pic_amount=10000
class_number=10
node_number=4
batchsize=100
np.random.seed(4)
w_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,size))
b_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,1))
w_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,node_number))
b_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,1))

def onenode(x_idx):
    x_changeToone = x_idx.reshape(size,batchsize)/256.0
    t_onenode = w_onenode @ x_changeToone + b_onenode
    y_onenode = 1/(1+np.exp(-t_onenode))
    print(y_onenode)
    return y_onenode

def twonode(y_nodeone):
    a_twonode = w_twonode @ y_nodeone + b_twonode
    alpha = a_twonode.max(axis=0,keepdims=True)
    y_twonode_exp = np.exp(a_twonode - alpha)
    y_twonode_sum = y_twonode_exp.sum(axis=0,keepdims=True)
    y_twonode = np.divide(y_twonode_exp,y_twonode_sum)
    print(y_twonode)
    print(y_twonode.shape)
    return y_twonode

def one_hot_vector(y):
    one_hot=np.zeros((class_number,batchsize))
    one_hot[y,np.arange(batchsize)]=1
    return one_hot

def exp_margin(y_onehot,y_pred):
    y_product = - y_onehot * np.log(y_pred)
    margin_exp = y_product.sum() / batchsize
    return margin_exp

learn_size=X.shape[0]
batchidx=np.random.choice(learn_size,batchsize,replace=False)
minibatch=X[batchidx]
y_onenode = onenode(minibatch)
y_twonode = twonode(y_onenode)
y_onehot = one_hot_vector(Y[batchidx])
exp_margin_value = exp_margin(y_onehot,y_twonode)
print("エントロピー誤差は",exp_margin_value,"です。")