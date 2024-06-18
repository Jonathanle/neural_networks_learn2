import pytest
import numpy as np
from math import tanh


from digit_classifier import Layer, Network 

# Define a pytest fixture for the setup


XOR_dataset = [[(0,0), 0], [(0, 1), 1], [(1, 0), 1], [(1,1), 0]]

@pytest.fixture(scope='module')
def setup_activation():
    activation_function = np.vectorize(tanh)
    
    return activation_function


@pytest.fixture(scope='function')
def setup_layer():
    activation_function = np.vectorize(tanh)
    layer = Layer(num_incoming_input=2, num_nodes=2)
    layer.set_weights(np.array([[2, 4], [6, 8]]))
    return layer

@pytest.fixture(scope='function')
def setup_network():
    layers_num_nodes = [2, 2, 2, 2]
    network = Network(layers_num_nodes)


    network._layers[0].set_weights(np.ones((2, 2)))
    network._layers[1].set_weights(np.ones((2, 2)))
    network._layers[2].set_weights(np.ones((2, 2)))

    return network 

def test_feedforward_network(setup_network, setup_activation):
    value = setup_network.forward(np.array([[3], [4]]))

    print(value.shape)
    assert value[0] - 0.958575872 < 0.001
    assert value[0] - 0.958575872 < 0.001
    
def test_gradient(setup_network, setup_activation):


    layers_num_nodes = [2, 2]
    network = Network(layers_num_nodes)


    network._layers[0].set_weights(np.ones((2, 2)))
   

    value = network.forward([[0.5], [0.8]])


    print(value)
    gradient = network.compute_gradient([[1], [1]])


    print(value - 1)
    print(gradient[0])
    print(gradient[1])

    assert True


def test_gradient_3_layer(setup_network, setup_activation):


    layers_num_nodes = [2, 2, 2, 2]
    network = Network(layers_num_nodes)


    network._layers[0].set_weights(np.ones((2, 2)))
    network._layers[1].set_weights(np.ones((2, 2)))
    network._layers[2].set_weights(np.ones((2, 2)))
   

    value = network.forward([[0.5], [0.8]])

    gradient = network.compute_gradient([[1], [1]])


    print(gradient[0])
    print(gradient[1])

    assert False
# Define the test function
def test_layer_forward(setup_layer, setup_activation):

    output = setup_layer.forward(np.array([[1], [1]]))
    print(setup_activation(np.array([[6], [14]])))
    assert np.array_equal(output, setup_activation(np.array([[6], [14]])))


def test_xor_training(): 
    layers_num_nodes = [2, 2, 2, 1]
    network = Network(layers_num_nodes)


    network._layers[0].set_weights(np.ones((2, 2)))
    network._layers[1].set_weights(np.ones((2, 2)))


    for i in range(0, 1):
        for j in range(0,4):
            X = np.array(XOR_dataset[j][0])
            X = X.reshape(1, -1)
            X = np.transpose(X)

            y = np.array(XOR_dataset[j][1])
            y = y.reshape(1, -1)
            y = np.transpose(y)
            network.train_1_data(X, y)

    
    print(network.forward(np.array([[1], [0]])))
    assert False


if __name__ == '__main__':
    pytest.main()