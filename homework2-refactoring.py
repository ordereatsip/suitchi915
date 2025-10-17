import numpy as np
import mnist
from network import NeuralNetwork # network.pyからクラスをインポート
from utils import to_one_hot_vector, calculate_cross_entropy # utils.pyから関数をインポート

# --- 設定値 (定数) ---
IMAGE_SIZE = 784
HIDDEN_NODES = 4
OUTPUT_CLASSES = 10
BATCH_SIZE = 100

def main():
    """メイン処理"""
    # 1. データの読み込み
    X_train = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
    Y_train = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
    
    # 2. ネットワークのインスタンスを作成
    model = NeuralNetwork(
        input_size=IMAGE_SIZE,
        hidden_size=HIDDEN_NODES,
        output_size=OUTPUT_CLASSES
    )
    
    # 3. ミニバッチの準備
    learn_size = X_train.shape[0]
    batch_indices = np.random.choice(learn_size, BATCH_SIZE, replace=False)
    x_minibatch = X_train[batch_indices]
    y_minibatch_labels = Y_train[batch_indices]
    
    # 4. 順伝播を実行して予測値を取得
    y_predictions = model.forward(x_minibatch)
    
    # 5. 正解ラベルをワンホットベクトルに変換
    y_true_one_hot = to_one_hot_vector(y_minibatch_labels, OUTPUT_CLASSES, BATCH_SIZE)
    
    # 6. 損失（クロスエントロピー誤差）を計算
    loss = calculate_cross_entropy(y_predictions, y_true_one_hot)

    print(f"クロスエントロピー誤差は {loss} です。")

if __name__ == '__main__':
    main()