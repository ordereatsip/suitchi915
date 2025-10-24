# network.py
"""
ニューラルネットワークのクラスを定義します。
(バイアスをランダムな正規分布で初期化するバージョン)
"""
import numpy as np
from utils import  softmax,relu,relu_derivative,dropout,dropout_test,dropout_derivative

class NeuralNetwork:
    """3層ニューラルネットワーク (入力層 - 隠れ層 - 出力層)"""

    def __init__(self, input_size, hidden_size, output_size):
        """重みとバイアスを初期化します"""
        np.random.seed(4) # 再現性のためにseedを固定
        
        # レイヤー1 (入力層 -> 隠れ層)
        self.w1 = np.random.normal(0, np.sqrt(1/input_size), (hidden_size, input_size))
        # ▼▼▼ バイアスを正規分布で初期化 ▼▼▼
        self.b1 = np.random.normal(0, np.sqrt(1/input_size), (hidden_size, 1))
        
        # レイヤー2 (隠れ層 -> 出力層)
        self.w2 = np.random.normal(0, np.sqrt(1/hidden_size), (output_size, hidden_size))
        # ▼▼▼ バイアスを正規分布で初期化 ▼▼▼
        self.b2 = np.random.normal(0, np.sqrt(1/hidden_size), (output_size, 1))
        
        self.gradients = {} # 逆伝播で計算した勾配を保持する
        self.cache = {}     # 順伝播の途中の値を保持する

    def forward(self, x_batch,drop_rate,rad_array,is_train):
        """順伝播 (x_batch は (batchsize, 784) の形状を想定)"""
        # 入力データを (784, batchsize) に変形・正規化
        x_processed = x_batch.reshape(-1, self.w1.shape[1]).T / 255.0
        # レイヤー1 (隠れ層)
        t1 = self.w1 @ x_processed + self.b1
        y1 = relu(t1)
        # ドロップアウト
        if(is_train):
            y1_dropped = dropout(y1,rad_array,drop_rate)
        else:
            y1_dropped = dropout_test(y1,drop_rate)
        # レイヤー2 (出力層)
        a2 = self.w2 @ y1_dropped + self.b2
        y2_pred = softmax(a2)
        
        # 逆伝播のために途中の値を保存
        self.cache = {'x_processed': x_processed, 'y1': y1}
        
        return y2_pred

    def backward(self, y_true_one_hot,y_pred,rad_array,drop_rate):
        """逆伝播 (バッチサイズで正規化済みの勾配を計算)"""
        batch_size = y_pred.shape[1]
        
        # 1. 出力層 (Softmax + CrossEntropy) の勾配
        delta_a2 = (y_pred - y_true_one_hot) / batch_size
        
        # 2. レイヤー2 (w2, b2) の勾配
        dw2 = delta_a2 @ self.cache['y1'].T
        db2 = np.sum(delta_a2, axis=1, keepdims=True)
        # 3. 隠れ層 (y1) への勾配
        delta_y1 = self.w2.T @ delta_a2
        
        #ドロップアウト
        delta_y1_dropped = dropout_derivative(delta_y1,rad_array,drop_rate)
        # 4. 隠れ層 (Relu) の勾配
        delta_t1 = delta_y1_dropped * relu_derivative(self.cache['y1'])
        
        # 5. レイヤー1 (w1, b1) の勾配
        dw1 = delta_t1 @ self.cache['x_processed'].T
        db1 = np.sum(delta_t1, axis=1, keepdims=True)
        
        self.gradients = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    def update(self, learning_rate):
        """勾配を使って重みとバイアスを更新"""
        self.w1 -= learning_rate * self.gradients['dw1']
        self.b1 -= learning_rate * self.gradients['db1']
        self.w2 -= learning_rate * self.gradients['dw2']
        self.b2 -= learning_rate * self.gradients['db2']

    def save_weights(self, file_path='weight_file.npz'):
        """重みとバイアスを .npz ファイルに保存"""
        np.savez(file_path,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2)
        print(f"重みを {file_path} に保存しました。")

    def load_weights(self, file_path='weight_file.npz'):
        """ .npz ファイルから重みとバイアスを読み込む"""
        try:
            data = np.load(file_path)
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            print(f"{file_path} から重みを読み込みました。")
        except FileNotFoundError:
            print(f"エラー: {file_path} が見つかりません。初期重みで続行します。")