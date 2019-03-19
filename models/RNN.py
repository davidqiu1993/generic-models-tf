"""
Copyright (C) 2019, Dicong Qiu.
"""

import tensorflow as tf
from . import common


class RNN(object):
  """
  Recurrent neural network.
  """

  def __init__(self, options={}):
    super(RNN, self).__init__()

    _options = {
      'cell': 'lstm',
      'units': 64
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

    cell = common.rnn_cells[_options['cell']]
    self.variables['cell'] = cell(_options['units'])


  def connect(self, X, seq_length):
    """
    Connect the network.
    """

    self.states = {}

    rnn_outs, rnn_state_last = tf.nn.dynamic_rnn(
      self.variables['cell'], X, sequence_length=seq_length, dtype=tf.float32)
    rnn_outs = tf.transpose(rnn_outs, [1, 0, 2]) # move the length dimension

    self.states['rnn_outs']       = rnn_outs
    self.states['rnn_state_last'] = rnn_state_last

    return rnn_outs, rnn_state_last
