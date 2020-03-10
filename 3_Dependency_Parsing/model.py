import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None,
                 dep_embeddings=None):
        super(ParserModel, self).__init__()

        self.config = config
        
        # These are the hyper-parameters for choosing how many embeddings to
        # encode in the input layer.  See the last paragraph of 3.1
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12

        # Copy the Embedding data that we'll be using in the model.  Note that the
        # model gets these in the constructor so that the embeddings can come
        # from anywhere (the model is agnostic to the source of the embeddings).
        self.word_embeddings = word_embeddings
        self.pos_embeddings =  pos_embeddings # TODO
        self.dep_embeddings = dep_embeddings
        
        # Create the first layer of the network that transform the input data
        # (consisting of embeddings of words, their corresponding POS tags, and
        # the arc labels) to the hidden layer raw outputs.

        # TODO
        self.hidden_layer1 = nn.Linear(config.embedding_dim * (n_w + n_p + n_d) , config.l1_hidden_size)
        self.hidden_layer2 = nn.Linear(config.l1_hidden_size, config.l2_hidden_size)
        # After the activation of the hidden layer, you'll be randomly zero-ing
        # out a percentage of the activations, which is a process known as
        # "Dropout".  Dropout helps the model avoid looking at the activation of
        # one particular neuron and be more robust.  (In essence, dropout is
        # turning the one network into an *ensemble* of networks).  Create a
        # Dropout layer here that we'll use later in the forward() call.

        # TODO
        self.drop_layer = nn.Dropout(p=config.keep_prob)
                
        # Create the output layer that maps the activation of the hidden layer to
        # the output classes (i.e., the valid transitions)
        
        # TODO
        self.output_layer = nn.Linear(config.l2_hidden_size, config.num_classes)

        # Initialize the weights of both layers
        self.init_weights()
        
    def init_weights(self):
        # initialize each layer's weights to be uniformly distributed within this
        # range of +/-initrange.  This initialization ensures the weights have something to
        # start with for computing gradient descent and generally leads to
        # faster convergence.
        initrange = 0.1
        nn.init.uniform_(self.hidden_layer1.weight,a=-initrange,b=initrange)
        nn.init.uniform_(self.hidden_layer2.weight,a=-initrange,b=initrange)
        nn.init.uniform_(self.output_layer.weight,a=-initrange,b=initrange)

        
    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):
        
        # Based on the IDs, look up the embeddings for each thing we need.  Note
        # that the indices are a list of which embeddings need to be returned.

        # TODO
        w_embeddings = self.word_embeddings(word_indices)
        p_embeddings = self.pos_embeddings(pos_indices)
        d_embeddings = self.dep_embeddings(dep_indices)
        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):
        """
        Computes the next transition step (shift, reduce-left, reduce-right)
        based on the current state of the input.
        

        The indices here represent the words/pos/dependencies in the current
        context, which we'll need to turn into vectors.
        """
        
        # Look up the embeddings for this prediction.  Note that word_indices is
        # the list of certain words currently on the stack and buffer, rather
        # than a single word

        # TODO
        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)

        # Since we're converting lists of indices, we're getting a matrix back
        # out (each index becomes a vector).  We need to turn these into
        # single-dimensional vector (Flatten each of the embeddings into a
        # single dimension).  Note that the first dimension is the batch.  For
        # example, if we have a batch size of 2, 3 words per context, and 5
        # dimensions per embedding, word_embeddings should be tensor with size
        # (2,3,5).  We need it to be a tensor with size (2,15), which makes the
        # input just like that flat input vector you see in the network diagram.
        #
        # HINT: you don't need to copy data here, only reshape the tensor.
        # Functions like "view" (similar to numpy's "reshape" function will be
        # useful here.        

        # TODO
        w_embeddings,p_embeddings,d_embeddings = w_embeddings.view(-1, w_embeddings.shape[1] * w_embeddings.shape[2]),\
            p_embeddings.view(-1, p_embeddings.shape[1] * p_embeddings.shape[2]),\
            d_embeddings.view(-1, d_embeddings.shape[1] * d_embeddings.shape[2])
        
        # Compute the raw hidden layer activations from the concatentated input
        # embeddings.
        #
        # NOTE: if you're attempting the optional parts where you want to
        # compute separate weight matrices for each type of input, you'll need
        # do this step for each one!

        # TODO
        output = self.hidden_layer1(torch.cat((w_embeddings, p_embeddings, d_embeddings), 1))
        
        
        # Compute the cubic activation function here.
        #
        # NOTE: Pytorch doesn't have a cubic activation function in the library

        # TODO
        output = output ** 3

        # Now do dropout for final activations of the first hidden layer

        # TODO
        output = self.drop_layer(output)

        # Multiply the activation of the first hidden layer by the weights of
        # the second hidden layer and pass that through a ReLU non-linearity for
        # the final output activations.
        #
        # NOTE 1: this output does not need to be pushed through a softmax if
        # you're going to evaluate the output using the CrossEntropy loss
        # function, which will compute the softmax intrinsically as a part of
        # its optimization when computing the loss.

        # TODO
        output = self.hidden_layer2(output)
        F.relu_(output)

        output = self.output_layer(output)

        return output    

