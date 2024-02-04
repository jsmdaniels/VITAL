# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Spatio-Temporal Fusion Transformer Model.

Contains the SFT architecture and associated components. Defines functions
for training, evaluation and prediction using simple Pandas Dataframe inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import json
import os
import shutil

import pipeline_blocks.data_formatters.base
import pipeline_blocks.libs.utils as utils
import pipeline_blocks.preprocessing.dataparser as dataparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


# Layer definitions.
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

# Default input types.
InputTypes = pipeline_blocks.data_formatters.base.InputTypes

# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
  """Returns simple Keras linear layer.

  Args:
    size: Output size
    activation: Activation function to apply if required
    use_time_distributed: Whether to apply layer across time
    use_bias: Whether bias should be included in layer
  """
  linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
  if use_time_distributed:
    linear = tf.keras.layers.TimeDistributed(linear)
  return linear


def apply_mlp(inputs,
              hidden_size,
              output_size,
              output_activation=None,
              hidden_activation='tanh',
              use_time_distributed=False):
  """Applies simple feed-forward network to an input.

  Args:
    inputs: MLP inputs
    hidden_size: Hidden state size
    output_size: Output size of MLP
    output_activation: Activation function to apply on output
    hidden_activation: Activation function to apply on input
    use_time_distributed: Whether to apply across time

  Returns:
    Tensor for MLP outputs.
  """
  if use_time_distributed:
    hidden = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(
            inputs)
    return tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_size, activation=output_activation))(
            hidden)
  else:
    hidden = tf.keras.layers.Dense(
        hidden_size, activation=hidden_activation)(
            inputs)
    return tf.keras.layers.Dense(
        output_size, activation=output_activation)(
            hidden)


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
  """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

  if dropout_rate is not None:
    x = tf.keras.layers.Dropout(dropout_rate)(x)

  if use_time_distributed:
    activation_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
            x)
    gated_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
            x)
  else:
    activation_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation=activation)(
            x)
    gated_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation='sigmoid')(
            x)

  return tf.keras.layers.Multiply()([activation_layer,
                                     gated_layer]), gated_layer


def add_and_norm(x_list):
  """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
  tmp = Add()(x_list)
  tmp = LayerNorm()(tmp)
  return tmp


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
  """Applies the gated residual network (GRN) as defined in paper.

  Args:
    x: Network inputs
    hidden_layer_size: Internal state size
    output_size: Size of output layer
    dropout_rate: Dropout rate if dropout is applied
    use_time_distributed: Whether to apply network across time dimension
    additional_context: Additional context vector to use if relevant
    return_gate: Whether to return GLU gate for diagnostic purposes

  Returns:
    Tuple of tensors for: (GRN output, GLU gate)
  """

  # Setup skip connection
  if output_size is None:
    output_size = hidden_layer_size
    skip = x
  else:
    linear = Dense(output_size)
    if use_time_distributed:
      linear = tf.keras.layers.TimeDistributed(linear)
    skip = linear(x)

  # Apply feedforward network
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)(
          x)
  if additional_context is not None:
    hidden = hidden + linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed,
        use_bias=False)(
            additional_context)
  hidden = tf.keras.layers.Activation('elu')(hidden)
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)(
          hidden)

  gating_layer, gate = apply_gating_layer(
      hidden,
      output_size,
      dropout_rate=dropout_rate,
      use_time_distributed=use_time_distributed,
      activation=None)

  if return_gate:
    return add_and_norm([skip, gating_layer]), gate
  else:
    return add_and_norm([skip, gating_layer])


# Attention Components.
def get_decoder_mask(self_attn_inputs):
  """Returns causal mask to apply for self-attention layer.

  Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
  """
  len_s = tf.shape(self_attn_inputs)[1]
  bs = tf.shape(self_attn_inputs)[:1]
  mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
  return mask


class ScaledDotProductAttention():
  """Defines scaled dot product attention layer.

  Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
  """

  def __init__(self, attn_dropout=0.0):
    self.dropout = Dropout(attn_dropout)
    self.activation = Activation('softmax')

  def __call__(self, q, k, v, mask):
    """Applies scaled dot product attention.

    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
    # attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
    attn = tf.matmul(q, k, transpose_b=True)/temper
    if mask is not None:
      mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(
          mask)  # setting to infinity
      attn = Add()([attn, mmask])
    attn = self.activation(attn)
    attn = self.dropout(attn)
    # output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
    output = tf.matmul(attn, v)
    return output, attn
  



class InterpretableMultiHeadAttention():
  """Defines interpretable multi-head attention layer.

  Attributes:
    n_head: Number of heads
    d_k: Key/query dimensionality per head
    d_v: Value dimensionality
    dropout: Dropout rate to apply
    qs_layers: List of queries across heads
    ks_layers: List of keys across heads
    vs_layers: List of values across heads
    attention: Scaled dot product attention layer
    w_o: Output weight matrix to project internal state to the original TFT
      state size
  """

  def __init__(self, n_head, d_model, dropout):
    """Initialises layer.

    Args:
      n_head: Number of heads
      d_model: TFT state dimensionality
      dropout: Dropout discard rate
    """

    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout

    self.qs_layers = []
    self.ks_layers = []
    self.vs_layers = []

    # Use same value layer to facilitate interp
    vs_layer = Dense(d_v, use_bias=False)

    for _ in range(n_head):
      self.qs_layers.append(Dense(d_k, use_bias=False))
      self.ks_layers.append(Dense(d_k, use_bias=False))
      self.vs_layers.append(vs_layer)  # use same vs_layer

    self.attention = ScaledDotProductAttention()
    self.w_o = Dense(d_model, use_bias=False)

  def __call__(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.

    Using T to denote the number of time steps fed into the transformer.

    Args:
      q: Query tensor of shape=(?, T, d_model)
      k: Key of shape=(?, T, d_model)
      v: Values of shape=(?, T, d_model)
      mask: Masking if required with shape=(?, T, T)

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    n_head = self.n_head

    heads = []
    attns = []
    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask)

      head_dropout = Dropout(self.dropout)(head)
      heads.append(head_dropout)
      attns.append(attn)
    head = K.stack(heads) if n_head > 1 else heads[0]
    attn = K.stack(attns)

    outputs = K.mean(head, axis=0) if n_head > 1 else head
    outputs = self.w_o(outputs)
    outputs = Dropout(self.dropout)(outputs)  # output dropout

    return outputs, attn


class SFTDataCache(object):
  """Caches data for the SFT."""

  _data_cache = {}

  @classmethod
  def update(cls, data, key):
    """Updates cached data.

    Args:
      data: Source to update
      key: Key to dictionary location
    """
    cls._data_cache[key] = data

  @classmethod
  def get(cls, key):
    """Returns data stored at key location."""
    return cls._data_cache[key].copy()

  @classmethod
  def contains(cls, key):
    """Retuns boolean indicating whether key is present in cache."""

    return key in cls._data_cache



class SFTAttentionModule(object):
  """Defines Spatio-temporal Fusion Transforme Attention module.

  Attributes:
    name: Name of model
    model: Keras model for SFT
  """
  def __init__(self, attn):
    """Builds SFT from parameters.

    Args:
      params: Parameters to define SFT
      use_cudnn: Whether to use CUDNN GPU optimised LSTM
    """
    self.name = self.__class__.__name__

    # Extra components to store Tensorflow nodes for attention computations
    self._input_placeholder = None
    self._attention_components = attn

  
  def get_attention(self, df):
    """Computes TFT attention weights for a given dataset.

    Args:
      df: Input dataframe

    Returns:
        Dictionary of numpy arrays for temporal attention weights and variable
          selection weights, along with their identifiers and time indices
    """

    data = self._batch_data(df)
    inputs = data['inputs']
    identifiers = data['identifier']
    time = data['time']

    def get_batch_attention_weights(input_batch):
      """Returns weights for a given minibatch of data."""
      input_placeholder = self._input_placeholder
      attention_weights = {}
      for k in self._attention_components:
        attention_weight = tf.keras.backend.get_session().run(
            self._attention_components[k],
            {input_placeholder: input_batch.astype(np.float32)})
        attention_weights[k] = attention_weight
      return attention_weights

    # Compute number of batches
    batch_size = self.minibatch_size
    n = inputs.shape[0]
    num_batches = n // batch_size
    if n - (num_batches * batch_size) > 0:
      num_batches += 1

    # Split up inputs into batches
    batched_inputs = [
        inputs[i * batch_size:(i + 1) * batch_size, Ellipsis]
        for i in range(num_batches)
    ]

    # Get attention weights, while avoiding large memory increases
    attention_by_batch = [
        get_batch_attention_weights(batch) for batch in batched_inputs
    ]
    attention_weights = {}
    for k in self._attention_components:
      attention_weights[k] = []
      for batch_weights in attention_by_batch:
        attention_weights[k].append(batch_weights[k])

      if len(attention_weights[k][0].shape) == 4:
        tmp = np.concatenate(attention_weights[k], axis=1)
      else:
        tmp = np.concatenate(attention_weights[k], axis=0)

      del attention_weights[k]
      gc.collect()
      attention_weights[k] = tmp

    attention_weights['identifiers'] = identifiers[:, 0, 0]
    attention_weights['time'] = time[:, :, 0]

    return attention_weights
  
