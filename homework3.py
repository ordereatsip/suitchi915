import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
X2 = mnist.download_and_parse_mnist_file("C:/Users/Order/data/t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
Y2 = mnist.download_and_parse_mnist_file("C:/Users/Order/data/t10k-labels-idx1-ubyte.gz")
import matplotlib.pyplot as plt
from pylab import cm
size = 784
pic_amount=10000
class_number=10
node_number=80
batchsize=100
learning_rate = 0.01
learn_size=X.shape[0]
test_size=X2.shape[0]
or_initilize = input("重みを初期化しますか？ (y/n): ")
if or_initilize == 'y':
    data = np.load('weight_file.npz')
    w_onenode = data['w1']
    b_onenode = data['b1']
    w_twonode = data['w2']
    b_twonode = data['b2']
else:
    w_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,size))
    b_onenode = np.random.normal(0,(np.sqrt(1/size)),(node_number,1))
    w_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,node_number))
    b_twonode = np.random.normal(0,(np.sqrt(1/node_number)),(class_number,1))
def onenode(x_noramalized):
    t_onenode = w_onenode @ x_noramalized + b_onenode
    y_onenode = 1/(1+np.exp(-t_onenode))
    return y_onenode

def twonode(y_nodeone):
    a_twonode = w_twonode @ y_nodeone + b_twonode
    alpha = a_twonode.max(axis=0,keepdims=True)
    y_twonode_exp = np.exp(a_twonode - alpha)
    y_twonode_sum = y_twonode_exp.sum(axis=0,keepdims=True)
    y_twonode = np.divide(y_twonode_exp,y_twonode_sum)
    return y_twonode

def one_hot_vector(y):
    one_hot=np.zeros((class_number,batchsize))
    one_hot[y,np.arange(batchsize)]=1
    return one_hot

def exp_margin(y_onehot,y_pred):
    y_product = - y_onehot * np.log(y_pred)
    margin_exp = y_product.sum() / batchsize
    return margin_exp

def back_propagation_softmax(y_onehot,y_pred):
    delta_exp_a = (y_pred-y_onehot)/batchsize
    return delta_exp_a

def back_propagation_x(delta_exp_y,w_node):
    delta_exp_x = w_node.T @ delta_exp_y
    return delta_exp_x

def back_propagation_w(delta_exp_y,x):
    delta_exp_w = delta_exp_y @ x.T
    return delta_exp_w

def back_propagation_b(delta_exp_y):
    delta_exp_b = delta_exp_y.sum(axis = 1,keepdims=True)
    return delta_exp_b

def back_propagation_sigmoid(y,delta_exp_y):
    delta_exp_sigmoid =  delta_exp_y *(1-y) * y 
    return delta_exp_sigmoid

epochs = []
train_accuracies = []
test_accuracies = []

for j in range(100):
    exp_margin_value=0
    y_right_train=0
    y_right_test=0
    for i in range(learn_size//batchsize):
        batchidx = np.random.choice(learn_size, size=batchsize, replace=False)
        minibatch=X[batchidx]
        right_number=Y[batchidx]
        x_normalized = minibatch.reshape(batchsize,size).T /256.0
        y_onenode = onenode(x_normalized)
        y_twonode = twonode(y_onenode)
        y_max = np.argmax(y_twonode,axis=0)
        y_onehot = one_hot_vector(right_number)
        y_right_train += (y_max == right_number).sum()
        exp_margin_value += exp_margin(y_onehot,y_twonode)
        delta_exp_a = back_propagation_softmax(y_onehot,y_twonode)
        delta_exp_x2 = back_propagation_x(delta_exp_a,w_twonode)
        delta_exp_w2 = back_propagation_w(delta_exp_a,y_onenode)
        delta_exp_b2 = back_propagation_b(delta_exp_a)
        delta_exp_sigmoid = back_propagation_sigmoid(y_onenode,delta_exp_x2)
        delta_exp__w1 = back_propagation_w(delta_exp_sigmoid,x_normalized)
        delta_exp_b1 = back_propagation_b(delta_exp_sigmoid)
        w_twonode = w_twonode - learning_rate * delta_exp_w2
        b_twonode = b_twonode - learning_rate * delta_exp_b2
        w_onenode = w_onenode - learning_rate * delta_exp__w1
        b_onenode = b_onenode - learning_rate * delta_exp_b1
    exp_margin_average = exp_margin_value / (learn_size/batchsize)
    y_acurate = y_right_train / learn_size
    for i in range(test_size//batchsize):
        batchidx = np.random.choice(test_size, size=batchsize, replace=False)
        minibatch =X2[batchidx]
        right_number=Y2[batchidx]
        x_normalized = minibatch.reshape(batchsize,size).T /256.0
        y_onenode = onenode(x_normalized)
        y_twonode = twonode(y_onenode)
        y_max = np.argmax(y_twonode,axis=0)
        y_right_test += (y_max == right_number).sum()
    y_acurate_test = y_right_test / test_size
    print("エントロピー誤差の平均値:",exp_margin_average,"\n正解率:",y_acurate,"\nテストデータに対する正解率:",y_acurate_test)
    epochs.append(j + 1)
    train_accuracies.append(y_acurate)
    test_accuracies.append(y_acurate_test)

np.save('weight_file.npz',
        w1=w_onenode, 
         b1=b_onenode, 
         w2=w_twonode, 
         b2=b_twonode)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='訓練データの正解率 (Train Accuracy)')
plt.plot(epochs, test_accuracies, label='テストデータの正解率 (Test Accuracy)', linestyle='--')
plt.title('エポック毎の正解率の推移 (Accuracy over Epochs)')
plt.xlabel('エポック数 (Epoch)')
plt.ylabel('正解率 (Accuracy)')
plt.legend() 
plt.grid(True) 
plt.show()