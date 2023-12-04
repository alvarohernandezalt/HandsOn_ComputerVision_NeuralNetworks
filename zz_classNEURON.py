import numpy as np

class Neuron(object):
  """A simple feed-forward artificial neuron.
  Args:
    num_inputs (int): The input vector size / number of input values.
    activation_fn (callable): The activation function
  Attributes:
    W (ndarray): The weight values for each input.
    b (float): The bias value, added to the weighted sum.
    activation_fn (callable): The activation function.
  """
  def __init__(self, num_inputs, activation_fn()):
    super().__init__()
    # Randomly initializing the weight vector and bias value
    self.W = np.random.rand(num_inputs)
    self.b = np.random.rand(1)
    self.activation_fn = activation_fn
  
  def forward(self, x):
    """Forward the input signal through the neuron."""
    z = np.dot(x, self.W) + self.b
    return self.activation_fn(z)
  