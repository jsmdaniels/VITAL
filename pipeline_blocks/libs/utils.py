import os
import pathlib

import tensorflow as tf
import numpy as np
from math import pi

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

"""
---> GP utils by Joseph Futoma
"""
def rmse_error(y_true, y_pred):
    error =  K.abs(y_true - y_pred) * (120)
    return K.mean(error)

def RBF_kernel(length, x1, x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.pow(x1-x2,2.0)/length)
    return K

def OU_kernel(length, x1, x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.abs(x1-x2)/length)
    return K

def RatQuad_kernel(length, alpha, x1, x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.pow(1 + (tf.pow(x1-x2,2.0)/(length*alpha)),-alpha)
    return K

def ExpSineSquared_kernel(length, period, x1, x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-2.*tf.pow(tf.sin(tf.abs(x1-x2)*(pi/period)),2.0)/length)
    return K



def dot(x,y):
    """ dot product of two vectors """
    return tf.reduce_sum(tf.multiply(x,y))

def CG(A,b):
    """ Conjugate gradient, to get solution x = A^-1 * b,
    can be faster than using the Cholesky for large scale problems
    """
    b = tf.reshape(b,[-1])
    n = tf.shape(A)[0]
    x = tf.zeros([n])
    r_ = b
    p = r_

    #These settings are somewhat arbitrary
    #You might want to test sensitivity to these
    CG_EPS = tf.cast(n/1000,"float")
    MAX_ITER = tf.div(n,250) + 3

    def cond(i,x,r,p):
        return tf.logical_and(i < MAX_ITER, tf.norm(r) > CG_EPS)

    def body(i,x,r_,p):
        p_vec = tf.reshape(p,[-1,1])
        Ap = tf.reshape(tf.matmul(A,p_vec),[-1]) #make a vector

        alpha = dot(r_,r_)/dot(p,Ap)
        x = x + alpha*p
        r = r_ - alpha*Ap
        beta = dot(r,r)/dot(r_,r_)
        p = r + beta*p

        return i+1,x,r,p

    i = tf.constant(0)
    i,x,r,p = tf.while_loop(cond,body,loop_vars=[i,x,r_,p])

    return tf.reshape(x,[-1,1])

def Lanczos(Sigma_func,b):
    """ Lanczos method to approximate Sigma^1/2 * b, with b random vec

    Note: this only gives you a single draw though, which may not be ideal.

    Inputs:
        Sigma_func: function to premultiply a vector by Sigma, which you
            might not want to explicitly construct if it's huge.
        b: random vector of N(0,1)'

    Returns:
        random vector approximately equal to Sigma^1/2 * b
    """
    n = tf.shape(b)[0]
    k = tf.div(n,500) + 3 #this many Lanczos iterations

    betas = tf.zeros(1)
    alphas = tf.zeros(0)
    D = tf.zeros((n,1))

    b_norm = tf.norm(b)
    D = tf.concat([D,tf.reshape(b/b_norm,[-1,1])],1)

    def cond(j,alphas,betas,D):
        return j < k+1

    def body(j,alphas,betas,D):
        d_j = tf.slice(D,[0,j],[-1,1])
        d = Sigma_func(d_j) - tf.slice(betas,[j-1],[1])*tf.slice(D,[0,j-1],[-1,1])
        alphas = tf.concat([alphas,[dot(d_j,d)]],0)
        d = d - tf.slice(alphas,[j-1],[1])*d_j
        betas = tf.concat([betas,[tf.norm(d)]],0)
        D = tf.concat([D,d/tf.slice(betas,[j],[1])],1)
        return j+1,alphas,betas,D

    j = tf.constant(1)
    j,alphas,betas,D = tf.while_loop(cond,body,loop_vars=[j,alphas,betas,D],
        shape_invariants=[j.get_shape(),tf.TensorShape([None]),
                          tf.TensorShape([None]),tf.TensorShape([None,None])])

    betas_ = tf.diag(tf.slice(betas,[1],[k-1]))
    D_ = tf.slice(D,[0,1],[-1,k])

    #build out tridiagonal H: alphas_1:k on main, betas_2:k on off
    H = tf.diag(alphas) + tf.pad(betas_,[[1,0],[0,1]]) + tf.pad(betas_,[[0,1],[1,0]])

    e,v = tf.self_adjoint_eig(H)
    e_pos = tf.maximum(0.0,e)+1e-6 #make sure positive definite
    e_sqrt = tf.diag(tf.sqrt(e_pos))
    sq_H = tf.matmul(v,tf.matmul(e_sqrt,tf.transpose(v)))

    out = b_norm*tf.matmul(D_,sq_H)
    return tf.slice(out,[0,0],[-1,1]) #grab last column = *e_1

"""
---> TensorFlow utils by Bryan Lim
"""
# Generic.
def get_single_col_by_input_type(input_type, column_definition):
  """Returns name of single column.

  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

  l = [tup[0] for tup in column_definition if tup[2] == input_type]

  if len(l) != 1:
    raise ValueError('Invalid number of columns for {}'.format(input_type))

  return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
  """Extracts the names of columns that correspond to a define data_type.

  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude

  Returns:
    List of names for columns with data type specified.
  """
  return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
  ]

  # Loss functions.
def tensorflow_quantile_loss(y_true, y_pred, quantile):
  """Computes quantile loss for tensorflow.

  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper

  Args:
    y_true: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Tensor for quantile loss.
  """

  # Checks quantile
  if quantile < 0 or quantile > 1:
    raise ValueError(
        'Illegal quantile value={}! Values should be between 0 and 1.'.format(
            quantile))

  prediction_underflow = y_true - y_pred
  q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
      1. - quantile) * tf.maximum(-prediction_underflow, 0.)

  return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y_true, y_pred, quantile):
  """Computes normalised quantile loss for numpy arrays.

  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.

  Args:
    y_true: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Float for normalised quantile loss.
  """
  prediction_underflow = y_true - y_pred
  weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

  quantile_loss = weighted_errors.mean()
  normaliser = y_true.abs().mean()

  return 2 * quantile_loss / normaliser

# OS related functions.
def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
  """Creates tensorflow config for graphs to run on CPU or GPU.

  Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
  GPU machines.

  Args:
    tf_device: 'cpu' or 'gpu'
    gpu_id: GPU ID to use if relevant

  Returns:
    Tensorflow config.
  """

  if tf_device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
    tf_config = tf.ConfigProto(
        log_device_placement=False, device_count={'GPU': 0})

  else:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print('Selecting GPU ID={}'.format(gpu_id))

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

  return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
  """Saves Tensorflow graph to checkpoint.

  Saves all trainiable variables under a given variable scope to checkpoint.

  Args:
    tf_session: Session containing graph
    model_folder: Folder to save models
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope containing variables to save
  """
  # Save model
  if scope is None:
    saver = tf.train.Saver()
  else:
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

  save_path = saver.save(tf_session,
                         os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
  print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
  """Loads Tensorflow graph from checkpoint.

  Args:
    tf_session: Session to load graph into
    model_folder: Folder containing serialised model
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope to use.
    verbose: Whether to print additional debugging information.
  """
  # Load model proper
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print('Loading model from {0}'.format(load_path))

  print_weights_in_checkpoint(model_folder, cp_name)

  initial_vars = set(
      [v.name for v in tf.get_default_graph().as_graph_def().node])

  # Saver
  if scope is None:
    saver = tf.train.Saver()
  else:
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
  # Load
  saver.restore(tf_session, load_path)
  all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

  if verbose:
    print('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
    print('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
    print('All {0}'.format(','.join(all_vars)))

  print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
  """Prints all weights in Tensorflow checkpoint.

  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint

  Returns:

  """
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print_tensors_in_checkpoint_file(
      file_name=load_path,
      tensor_name='',
      all_tensors=True,
      all_tensor_names=True)
