

import numpy as np
from tqdm import tqdm
import pickle
import tensorflow as tf
import joblib


from digit_classifier import Layer, Network


import matplotlib.pyplot as plt
import numpy as np







XOR_dataset = [[(0,0), 0], [(0, 1), 1], [(1, 0), 1], [(1,1), 0]]


def process_X_set(dataset):
    size = dataset.shape[0]
    unravel_size = dataset.shape[1] * dataset.shape[2]

    x_new = np.zeros((size, unravel_size))



    # TODO: Optimize this thing for later.
    for i in range(size):
        x_new[i] = dataset[i].ravel()

    return x_new
def process_y_set(dataset):
    size = dataset.shape[0]
    
    y_new = np.full((size, 10), -1)


    for i in range(size): 
        y_new[i][dataset[i]] = 1

    return y_new

    
def load_mnist_dataset():
    mnist = tf.keras.datasets.mnist

    # Split the data into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # how to parallel preprocess data
    x_train = process_X_set(x_train)
    x_test = process_X_set(x_test)

    y_train = process_y_set(y_train) 
    y_test = process_y_set(y_test)


    return (x_train, y_train), (x_test, y_test)

def process_1d_array(X):
    X = np.array(X)
    X = X.reshape(1, -1)
    X = np.transpose(X)

    return X 

    
def train_mnist_training(): 
    layers_num_nodes = [784, 256, 256, 256, 10]
    network = Network(layers_num_nodes)


    dataset = load_mnist_dataset()

    x_train = dataset[0][0]
    y_train = dataset[0][1]




    for i in tqdm(range(0, 60000)):
        
        X = np.array(x_train[i])
        X = X.reshape(1, -1)
        X = np.transpose(X)

        y = np.array(y_train[i])
        y = y.reshape(1, -1)
        y = np.transpose(y)


        network.train_1_data(X, y)

   

    print("saving model to model.pkl")
    joblib.dump(network, "model.pkl")

    
    return;

    #TODO: Create an initialization mechanism.

def visualize_dataset(image): 
    # Create a 28x28 array with values from 0 to 255 (example)


    # Plot the array as an image
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()  # Optional: adds a colorbar to the side
    plt.title("28x28 Image")
    plt.axis('off')  # Hide the axis
    plt.show()
def index_of_largest(arr):
    """
    Returns the index of the largest number in the array.

    Parameters:
    - arr: list or numpy array of numbers

    Returns:
    - int: index of the largest number
    """
    if len(arr) == 0:
        raise ValueError("The array is empty.")
    
    largest_index = 0
    largest_value = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] > largest_value:
            largest_value = arr[i]
            largest_index = i
            
    return largest_index


def test_mnist_training():
    dataset = load_mnist_dataset()

    x_test = dataset[1][0]
    y_test = dataset[1][1]

    y_pred = []


    print("Loading model pkl file...")
    data = joblib.load("model.pkl")


    total_cost = 0
    correct = 0

    for x, y in tqdm(zip(x_test, y_test)):
        X = np.array(x)
        X = X.reshape(1, -1)
        X = np.transpose(X)

        y = np.array(y)
        y = y.reshape(1, -1)
        y = np.transpose(y)


        y_pred_point = data.forward(X)
        y_pred.append(y_pred_point)

        

        square_cost = np.sum(1 / y_pred_point.shape[0] * (y - y_pred_point)**2)

        # convert to 1d lists
        if index_of_largest(y.tolist()) == index_of_largest(y_pred_point.tolist()):
            correct += 1

        #print(1 / result.shape[0] * (y - y_pred)**2)
       
        total_cost += square_cost





    print(np.transpose(y_pred[100]))
    print(y_test[100])
    print(f"Average Loss: {total_cost / y_test.shape[0]}")
    print(f"Accuracy: {correct / y_test.shape[0]}")

    #visualize_dataset(y_test[0])

    


   




def test_xor_training(): 
    layers_num_nodes = [2, 2, 1]
    network = Network(layers_num_nodes)

    
    network._layers[0].set_weights(np.array([[-0.5, 0.5], [-0.5, 0.5]], dtype='float16'))
    network._layers[1].set_weights(np.array([[-0.5, 0.5]], dtype='float16'))

    
    network._layers[0].set_weights(np.array([[-0.5, 0.5], [0.5, -0.5]], dtype='float16'))
    network._layers[1].set_weights(np.array([[-0.5, 0.5]], dtype='float16'))

    

    
    for i in tqdm(range(0, 1000)):
        for j in range(0,4):
            X = np.array(XOR_dataset[j][0])
            X = X.reshape(1, -1)
            X = np.transpose(X)

            y = np.array(XOR_dataset[j][1])
            y = y.reshape(1, -1)
            y = np.transpose(y)


        
            network.train_1_data(X, y)
    

    
    #with open('model.pkl', 'rb') as file:
        #network = pickle.load(file)
    
    print(network.forward(np.array([[1], [0]])))
    print(network.forward(np.array([[0], [1]])))
    print(network.forward(np.array([[1], [1]])))
    print(network.forward(np.array([[0], [0]])))



    with open("model.pkl", "wb") as file:
        pickle.dump(network, file)

    

if __name__ == "__main__":
    #test_xor_training()
    train_mnist_training()
    test_mnist_training() 