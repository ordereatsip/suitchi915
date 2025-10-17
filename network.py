import numpy as np
from utils import sigmoid, softmax # utils.pyから関数をインポート

class NeuralNetwork:
    """3層ニューラルネットワークを表すクラス"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """ネットワークの重みとバイアスを初期化します"""
        
        # レイヤー1 (入力層 -> 隠れ層)
        self.w1 = np.random.normal(0, np.sqrt(1/input_size), (hidden_size, input_size))
        self.b1 = np.random.normal(0, np.sqrt(1/input_size), (hidden_size, 1))
        
        # レイヤー2 (隠れ層 -> 出力層)
        self.w2 = np.random.normal(0, np.sqrt(1/hidden_size), (output_size, hidden_size))
        self.b2 = np.random.normal(0, np.sqrt(1/hidden_size), (output_size, 1))
        
    def forward(self, x_batch, normalization_factor=255.0):
        """順伝播処理を行い、予測結果を返します"""
        x_processed = x_batch.reshape(-1, self.w1.shape[1]).T / normalization_factor
        
        t1 = self.w1 @ x_processed + self.b1
        y1 = sigmoid(t1) # インポートしたsigmoidを使用
        
        a2 = self.w2 @ y1 + self.b2
        y2_pred = softmax(a2) # インポートしたsoftmaxを使用
        
        return y2_pred