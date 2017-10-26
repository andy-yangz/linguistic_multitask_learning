"""
https://raw.githubusercontent.com/bplank/bilstm-aux/master/src/lib/mnnl.py
"""
import dynet
import numpy as np

import sys

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")

class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict_sequence(self, inputs):
        return [self.network_builder(x) for x in inputs]


class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return s_init.transduce(inputs)

class BiRNNSequencePredictor(SequencePredictor):
    """ a bidirectional RNN (LSTM/GRU) """
    def __init__(self, f_builder, b_builder):
        self.f_builder = f_builder
        self.b_builder = b_builder

    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.f_builder.initial_state()
        b_init = self.b_builder.initial_state()
        forward_sequence = f_init.transduce(f_inputs)
        backward_sequence = b_init.transduce(reversed(b_inputs))
        return forward_sequence, backward_sequence 
        

class Layer:
    """ Class for affine layer transformation or two-layer MLP """
    def __init__(self, model, in_dim, output_dim, activation=dynet.rectify):
        # if mlp > 0, add a hidden layer of that dimension
        self.act = activation
        self.W = model.add_parameters((output_dim, in_dim))
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)
        return self.act(W*x + b)
