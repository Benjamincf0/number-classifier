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
        s = self.__call__(x)  # Softmax output vector
        jacobian = np.diag(s) - np.outer(s, s)
        return jacobian

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
relu = reluClass()
softmax = softmaxClass()


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

        np.random.seed(SEED)
        # INITALIZING WEIGHTS
        weights = []
        biases = []
        # Adding weights to numpy array
        for i in range(1, len(dim)):
            # Initializing weights according to He Init.
            weights.append(np.sqrt(2 / dim[i-1])*np.random.randn(dim[i], dim[i-1]))
            # Initing biases to 0
            biases.append(np.zeros((dim[i], 1)))
        # Adding weights as an an attribute to this instance of Model
        self.weights = weights
        # Adding biases as an an attribute to this instance of Model
        self.biases = biases

        # Activations
        if not self.activation_functions:
            self.activation_functions = [sigmoid]*(len(self.dim) - 1)
            # fns.append(softmax)

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
        L_i = input_layer.reshape(-1, 1)
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

    def train(self, train_x, train_y, test_x, test_y, batch_size=None, learning_rate = 0.05, verbose = False):
        if not batch_size:
            batch_size = 200
        batch_index = 0
        sample_index = 0
        epoch_index = 0
        grad_w = []
        grad_b = []
        prev_accuracy = -1
        accuracy = accuracy = self.test(test_x, test_y)
        cost = 1
        prev_cost = 2
        gradient_w = [0]*(len(self.dim) - 1)
        print(f"train_x.shape[0] {train_x.shape[0]}\nbatch_size {batch_size}")
        cost = 0

        # Training loop
        while True:
            # y_sample = self.forward_prop(train_x[sample_index])
            y_label = np.zeros(self.dim[-1])
            y_label[train_y[sample_index]] = 1
            # cost += (y_label - y_sample)**2
            
            L_i = train_x[sample_index]
            activations = [L_i]
            fns = self.activation_functions

            # Forward prop, saving all activations
            for i in range(len(self.dim) - 1):
                L_i = fns[i]((np.dot(self.weights[i], L_i)) + self.biases[i])
                activations.append(L_i)
            # activations.append(y_label)
            
            # Computing cost
            cost += np.sum((y_label - activations[-1])**2)

            #Backward propagation to find gradient
            dL = 2*(y_label - activations[-1])*activations[-1]*(1-activations[-1])
            dL_list = [dL]
            # print(len(activations), activations[0].shape, activations[1].shape, activations[2].shape, activations[3].shape)
            for l in range(len(self.dim)-1)[::-1]:
                # grad_w_l = 2*activations[l-1]*(activations[l] - y_label)*softmax.prime(((np.dot(self.weights[l], activations[l-1])) + self.biases[l]))
                dL = np.dot(dL_list[-1], self.weights[l])*activations[l]*(1-activations[l])
                dL_list.append(dL)
                # grad_w.append(grad_w_l)
            
            # Compute gradients for weights and biases and do parameters - gradient
            sample_grad = []
            for l in range(len(self.dim)-1):
                delta = - np.dot(dL_list[-l - 1], activations[l])*learning_rate
                sample_grad.append(delta)
                gradient_w[l] += delta
                # self.biases[l] -= dL_list[l - 1]

            # Check if Batch is complete
            if (sample_index + 1) % batch_size == 0:
                batch_index += 1
                #Gradient descent
                for i in range(len(gradient_w)):
                    self.weights[i] -= gradient_w[i] / batch_size
                gradient_w = [0]*(len(self.dim) - 1)
                cost /= batch_size
                # Verbose
                sys.stdout.write(f"\rBatch: {batch_index}   Epoch: {epoch_index}   Accuracy: {accuracy}   Cost: {cost}")

            # Check if Epoch is complete
            if sample_index + 1 == train_x.shape[0]:
                epoch_index += 1
                accuracy = self.test(test_x, test_y)
                sys.stdout.write(f"\rBatch: {batch_index}   Epoch: {epoch_index}   Accuracy: {accuracy}   Cost: {cost}")
                if epoch_index == 200 or abs(prev_accuracy - accuracy) < 1:
                    print('\nDone!     ')
                    return {
                        'batch_index': batch_index,
                        'epoch_index': epoch_index,
                        'activations': len(activations),
                        # 'example' : f"y_sample: {y_sample}\ny_label: {y_label}",
                    }
                sample_index = 0

            sample_index += 1
                
    def test(self, test_x, test_y):
        """Tests model's accuracy on training set and returns the ratio of correct inferences over the total number of inferences

        Args:
            test_x (np.ndarray): inputs
            test_y (np.ndarray): labels

        Returns:
            double: ratio of correct/total inferences 
        """
        correct_inferences = 0
        for i in range(test_x.shape[0]):
            if np.argmax(self.forward_prop(test_x[i])) ==  test_y[i]:
                correct_inferences += 1
        print('\nTest Complete')
        return correct_inferences / test_x.shape[0]

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
    print(f'Here is the output:\n{my_model.forward_prop(np.array([0.5]*784))}')

#TESTINNGGGG
if __name__ == '__main__':
    main()