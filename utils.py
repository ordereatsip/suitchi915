# utils.py
"""
ニューラルネットワークで使用する補助的な関数群を定義します。
"""
import numpy as np
import mnist

def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    """シグモイド関数の微分 (y = sigmoid(x) の出力)"""
    return y * (1.0 - y)

def softmax(x):
    """ソフトマックス関数 (数値安定版)"""
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def input_normalization(x,batch_size,gamma,beta):
    epsilon = 1e-7
    average_batch = np.sum(x, axis=0) / batch_size
    variance = np.sum((x-average_batch)**2 ,axis=0) /batch_size
    x_normalization = (x-average_batch) / ((variance + epsilon)**(1/2))
    y_normalizetion = gamma * x_normalization + beta
    return (average_batch,variance,y_normalizetion,x_normalization)

def relu(x):
    return np.maximum(x,0)

def relu_derivative(y):
    return (y>0)

def dropout(x,rad_array,rate):
    return  (rad_array > rate).round().astype(int) * x

def dropout_test(x,rate):
    return x * (1 - rate)

def dropout_derivative (delta_exp_y,rad_array,rate): 
    return (rad_array > rate).round().astype(int) * delta_exp_y

def to_one_hot_vector(labels, num_classes, batch_size):
    """ラベルのバッチをワンホットベクトルに変換します"""
    one_hot = np.zeros((num_classes, batch_size))
    one_hot[labels, np.arange(batch_size)] = 1
    return one_hot

def calculate_cross_entropy(y_pred, y_true_one_hot):
    """クロスエントロピー誤差を計算します"""
    epsilon = 1e-7 # log(0) を防ぐ
    loss = -np.sum(y_true_one_hot * np.log(y_pred + epsilon)) / y_pred.shape[1]
    return loss

def calculate_accuracy(y_pred, y_true_labels):
    """予測 (確率) と正解ラベルから正解率を計算します"""
    y_pred_labels = np.argmax(y_pred, axis=0)
    accuracy = np.sum(y_pred_labels == y_true_labels) / len(y_true_labels)
    return accuracy

def load_mnist_data():
    """MNISTの学習データとテストデータを読み込みます"""
    print("MNISTデータを読み込んでいます...")
    X_train = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
    Y_train = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
    X_test = mnist.download_and_parse_mnist_file("C:/Users/Order/data/t10k-images-idx3-ubyte.gz")
    Y_test = mnist.download_and_parse_mnist_file("C:/Users/Order/data/t10k-labels-idx1-ubyte.gz")
    print("データの読み込みが完了しました。")
    return (X_train, Y_train), (X_test, Y_test)