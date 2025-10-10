import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
size = 784
class_number=10
node_number=4

def onenode(x_idx):
    x_changeToone = x_idx.reshape(size,1)/256
    np.random.seed(4)
    w_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,size))
    b_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,1))
    t_onenode = w_onenode @ x_changeToone + b_onenode
    y_onenode = 1/(1+np.exp(-t_onenode))
    return y_onenode

def twonode(y_nodeone):
    np.random.seed(4)
    w_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,node_number))
    b_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,1))
    a_twonode = w_twonode @ y_nodeone + b_twonode
    alpha = a_twonode.max()
    sum_aexp = 0
    for i in range(class_number):
        sum_aexp += np.exp(a_twonode[i]-alpha)
    y_twonode = np.exp(a_twonode-alpha)/sum_aexp
    return np.argmax(y_twonode)

idx = input("Please input an integer from 0 to 9999: ")
idx = int(idx)
X_idx = X[idx]
Y_onenode= onenode(X_idx)
Y_max = twonode(Y_onenode)
print("入力された画像の数字は",Y_max,"です。")
