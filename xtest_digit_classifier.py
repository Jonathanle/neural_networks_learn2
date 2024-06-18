import unittest
from digit_classifier import Layer
# Sample function to be tested

import numpy as np


class TestSampleFunction(unittest.TestCase):

    def setUp(self):

        self.layer = Layer(num_incoming_input=2, num_nodes = 2)
        self.layer.set_weights(np.array([[2, 4], [6, 8]]))
        
        
    def test_sample_function(self):
        output = np.array(self.layer.forward(np.array([[1], [1]])))

        print(output)

        
        self.assertTrue(np.array_equal(output, np.array([[6], [14]])))

if __name__ == '__main__':
    unittest.main()
