import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[42]
#plt.imshow(digit, cmap=plt.cm.binary)
#plt.show()
print(digit.shape)
#(28, 28)
# So a digit in this representation is a matrix 28 x 28
my_slice = train_images[10:100]
print(my_slice.shape)
#(90, 28, 28)