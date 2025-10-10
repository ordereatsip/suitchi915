import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("C:/Users/Order/data/train-labels-idx1-ubyte.gz")
import matplotlib.pyplot as plt
from pylab import cm
idx = 100
plt.imshow(X[idx], cmap=cm.gray)
plt.show()
print(Y[idx])
