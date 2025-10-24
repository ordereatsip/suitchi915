"""
ニューラルネットワークの学習と評価を実行するメインスクリプト。
"""
import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork
from utils import (load_mnist_data, to_one_hot_vector, 
                   calculate_cross_entropy, calculate_accuracy)

# --- 設定値 (定数) ---
INPUT_SIZE = 784
HIDDEN_NODES = 80
OUTPUT_CLASSES = 10
BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_FILE = 'weight_file.npz'

def main():
    (X_train, Y_train), (X_test, Y_test) = load_mnist_data()
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_NODES, OUTPUT_CLASSES)
    
    load_params_prompt = input("学習済みのパラメータをファイルから読み込みますか？ (y/n): ")
    
    if load_params_prompt.lower() == 'y':
        model.load_weights(WEIGHT_FILE)

    print(f"学習を開始します (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE})")
    
    epochs_list = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        total_train_loss = 0
        total_train_accuracy = 0
        
        permutation = np.random.permutation(train_size)
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        for i in range(0, train_size, BATCH_SIZE):
            x_batch = X_train_shuffled[i : i + BATCH_SIZE]
            y_batch_labels = Y_train_shuffled[i : i + BATCH_SIZE]
            
            current_batch_size = x_batch.shape[0]
            if current_batch_size != BATCH_SIZE:
                continue
            
            y_batch_one_hot = to_one_hot_vector(y_batch_labels, OUTPUT_CLASSES, current_batch_size)
            
            y_pred = model.forward(x_batch)
            total_train_loss += calculate_cross_entropy(y_pred, y_batch_one_hot)
            total_train_accuracy += calculate_accuracy(y_pred, y_batch_labels)
            
            model.backward(y_batch_one_hot, y_pred)
            model.update(LEARNING_RATE)
        
        avg_train_loss = total_train_loss / (train_size // BATCH_SIZE)
        avg_train_accuracy = total_train_accuracy / (train_size // BATCH_SIZE)
        
        total_test_accuracy = 0
        for i in range(0, test_size, BATCH_SIZE):
            x_test_batch = X_test[i : i + BATCH_SIZE]
            y_test_batch_labels = Y_test[i : i + BATCH_SIZE]
            
            current_batch_size = x_test_batch.shape[0]
            if current_batch_size != BATCH_SIZE:
                continue

            y_test_pred = model.forward(x_test_batch)
            total_test_accuracy += calculate_accuracy(y_test_pred, y_test_batch_labels)
        
        avg_test_accuracy = total_test_accuracy / (test_size // BATCH_SIZE)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_accuracy:.4f}, "
              f"Test Acc: {avg_test_accuracy:.4f}")
        
        epochs_list.append(epoch + 1)
        train_accuracies.append(avg_train_accuracy)
        test_accuracies.append(avg_test_accuracy)

    model.save_weights(WEIGHT_FILE)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_accuracies, label='訓練データの正解率 (Train Accuracy)')
    plt.plot(epochs_list, test_accuracies, label='テストデータの正解率 (Test Accuracy)', linestyle='--')
    plt.title('エポック毎の正解率の推移 (Accuracy over Epochs)')
    plt.xlabel('エポック数 (Epoch)')
    plt.ylabel('正解率 (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()