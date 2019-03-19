"""
Copyright (C) 2019, Dicong Qiu.
"""

import tensorflow as tf
from . import common


class CNN(object):
  """
  Convolutional neural network.
  """

  def __init__(self, options={}):
    super(CNN, self).__init__()

    _options = {
      'layers': [
        ('conv', 32, [3, 3], 'relu'), # (filters, kernel, act)
        ('pool', [2, 2], 2) # (size, strides)
      ],
      'padding': 'same'
    }

    # load options
    for o in _options:
      if _options[o] is None and o not in options:
        raise Error('Missing required option %s' % (o))
      if o in options:
        _options[o] = options[o]

    self.options = _options

    # initialize variables
    self.variables = {}


  def connect(self, X):
    """
    Connect the network.
    """

    layers = self.options['layers']
    padding = self.options['padding']

    self.states = {}

    H = X
    layer_outputs = []
    for i in range(len(layers)):
      layer = layers[i]

      # connect layer
      if layer[0] == 'conv':
        H = tf.layers.conv2d(
          inputs=H,
          filters=layer[1],
          kernel_size=layer[2],
          padding=padding,
          activation=common.activations[layer[3]]
        )
      elif layer[0] == 'pool':
        H = tf.layers.max_pooling2d(
          inputs=H,
          pool_size=layer[1],
          strides=layer[2]
        )
      else:
        raise Error('Invalid layer type %s' % (layer[0]))

      # add to outputs
      layer_outputs.append(H)

    # add layer outputs to states
    self.states['layer_outputs'] = layer_outputs

    return layer_outputs[-1]
