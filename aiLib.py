import numpy as np
import sys
# import time

# Default activation functions
class reluClass:
    def __call__(self, x):
        return np.maximum(0, x)
    def prime(self, x):
        return np.where(x > 0, 1, 0)
    
class softmaxClass:
    def __call__(self, x):
        max = np.max(x)
        exp_x = np.exp(x - max)
        return exp_x / np.sum(exp_x)
    def prime(self, x):
        s = softmax(x)  # Softmax output vector
        jacobian = np.diag(s) - np.outer(s, s)
        return jacobian
    
relu = reluClass()


# def relu(x):
#     # Apply ReLU element-wise
#     # print('relu')
#     return np.maximum(0, x)

# def reluprime(x):
#     return np.where(x > 0, 1, 0)

# def softmax(x):
#     # print('softmax')
#     max = np.max(x)
#     exp_x = np.exp(x - max)
#     return exp_x / np.sum(exp_x)

# def softmaxprime(x):
#     s = softmax(x)  # Softmax output vector
#     jacobian = np.diag(s) - np.outer(s, s)
#     return jacobian

SEED = 32341564

class Model:
    '''
    The model class allows for the construction of a Neural Network, varying parameters.
    '''
    count_inst = 0

    def __init__(self, model_name =f'model_{count_inst + 1}', dim=[784, 20, 20, 11], activation_functions=None):
        self.model_name = model_name
        self.dim = dim
        self.activation_functions = activation_functions
        Model.count_inst += 1

        # INITALIZING WEIGHTS
        rng = np.random.seed(SEED)
        weights = []
        # INIT BIASES
        biases = []

        # Adding weights to numpy array
        for i in range(1, len(dim)):
            # Initializing weights according to He Init.
            weights.append(np.sqrt(2 / dim[i-1])*np.random.randn(dim[i], dim[i-1]))
            # Initing biases to 0
            biases.append(np.zeros(dim[i]))
        # Adding weights as an an attribute to this instance of Model
        self.weights = weights
        # Adding biases as an an attribute to this instance of Model
        self.biases = biases


    def __str__(self):
        output = f'\nNeural Network:\n'
        output += f"{'+---------------+            '*(len(self.dim) - 1)}+----------------+\n"
        output += '|  Input Layer  |            '
        for i in range(len(self.dim) - 2):
            output += f'| Hidden Layer {i+1}|            '
        output += '|  Output Layer  |\n'
        
        output += f'|  {self.dim[0]} Neurons  | ---------> '
        for i in range(len(self.dim)-2):
            neurons = self.dim[1:-1][i]
            if neurons < 10:
                output += f'|   {neurons} Neurons   | ---------> '
            elif neurons < 100:
                output += f'|  {neurons} Neurons   | ---------> '
            else:
                output += f'|  {neurons} Neurons  | ---------> '

        output += f'|   {self.dim[-1]} Neurons   |\n'

        output += f"{'+---------------+            '*(len(self.dim) - 1)}+----------------+\n"
        return output

    def forward_prop(self, input_layer):
        L_i = input_layer
        if not self.activation_functions:
            fns = [relu]*(len(self.dim) - 1)
            # fns.append(softmax)
        else:
            fns = self.activation_functions

        # print(f"L_i: {L_i}")
        for i in range(len(fns)):
            L_i = fns[i]((np.dot(self.weights[i], L_i)) + self.biases[i])
            # print(f"L_i: {L_i}")
        return L_i
    
    def backward_prop(self, gradient, learning_rate):
        # for i in reversed(range(len(self.dim))):
        #     for j in range(len(self.weights[i])):
        #         self.weights[i][j] - gradient*learning_rate
        #         self.biases[0]

        for l in range(len(self.dim-1))[::-1]:
            return 

    def train(self, train_X, train_Y, batch_size=None, learning_rate = 1, verbose = False):
        if not batch_size:
            batch_size = 100
        batch_index = 0
        sample_index = 0
        epoch_index = 0
        gradient = 0
        prev_accuracy = -1
        accuracy = 0
        cost = 1
        prev_cost = 2
        print(f"train_X.shape[0] {train_X.shape[0]}\nbatch_size {batch_size}")
        while True:
            # time.sleep(0.001)
            y_sample = self.forward_prop(train_X[sample_index])
            y_label = np.zeros(self.dim[-1])
            y_label[train_Y[sample_index]] = 1
            gradient += (y_label - y_sample)**2

            # Check if Batch is complete
            if (sample_index + 1) % batch_size == 0:
                sys.stdout.write(f"\rBatch: {batch_index+1}   Epoch: {epoch_index}   Accuracy: {accuracy}   Cost: {cost}")
                batch_index += 1
                # self.backward_prop(gradient/(sample_index + 1), learning_rate)

            # Check if Epoch is complete
            if sample_index + 1 == train_X.shape[0]:
                sys.stdout.write(f"\rBatch: {batch_index}   Epoch: {epoch_index+1}   Accuracy: {accuracy}   Cost: {cost}")
                if epoch_index + 1 == 10 or prev_accuracy == accuracy:
                    print('\nDone!     ')
                    return {
                        'batch_index': batch_index,
                        'epoch_index': epoch_index,
                        'example' : f"y_sample: {y_sample}\ny_label: {y_label}",
                    }
                epoch_index += 1
                sample_index = 0

            sample_index += 1
                


# When running this module as the main program
def main():
    my_model = Model()
    print(my_model)

    for i in range(len(my_model.weights)):
        if i+1 == len(my_model.weights):
            text = 'Output Layer'
        else:
            text = f'Hidden Layer {i+1}'
        print(f'{text}\nWeights: {my_model.weights[i].shape}\nBiases: {my_model.biases[i].shape}\nBiases dimension: {my_model.biases[i].ndim}\n')

    print('\nNOW RUNNING FORWARD PROP\n')
    print(f'Here is the output:\n{my_model.forward_prop([0.5]*784)}')

#TESTINNGGGG
if __name__ == '__main__':
    main()