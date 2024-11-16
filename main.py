import aiLib as ai
import numpy as np
from keras.datasets import mnist # type: ignore

(train_X, train_y), (test_X, test_y) = mnist.load_data()
# print(train_X.shape, train_y.shape)
# Convert to 2d arrays rather than 3d
train_x = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
test_x = np.reshape(test_X, (test_X.shape[0], test_X.shape[1]*test_X.shape[2]))

# print(test_x.shape[0], test_y.shape)
num_model = ai.Model()
print(num_model.train(train_x, train_y, test_x, test_y))

def main():
    """adfasdfa

    Args:
        a (integer): variable that does stuff
        b (heyyy): yayyy
    """

    

if __name__ == '__main__':
    main()