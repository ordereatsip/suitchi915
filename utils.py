import numpy as np

def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ソフトマックス関数"""
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def to_one_hot_vector(labels, num_classes, batch_size):
    """ラベルのバッチをワンホットベクトルに変換します"""
    one_hot = np.zeros((num_classes, batch_size))
    one_hot[labels, np.arange(batch_size)] = 1
    return one_hot

def calculate_cross_entropy(y_pred, y_true_one_hot):
    """クロスエントロピー誤差を計算します"""
    epsilon = 1e-7 # log(0)を防ぐための微小な値
    loss = -np.sum(y_true_one_hot * np.log(y_pred + epsilon)) / y_pred.shape[1]
    return loss