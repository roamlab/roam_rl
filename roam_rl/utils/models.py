import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.models import get_network_builder
from .config_utils import initfromconfig

class NetworkFn:

    def __init__(self, config, section):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        #pylint: disable=E1101
        return self._network_fn(*args, **kwargs)

class MLP(NetworkFn):

    def __init__(self, config, section):
        network_args = {}
        if config.has_option(section, 'num_layers'):
            network_args['num_layers'] = config.getint(section, 'num_layers')
        if config.has_option(section, 'num_hidden'):
            network_args['num_hidden'] = config.getint(section, 'num_hidden')
        if config.has_option(section, 'activation'):
            network_args['activation'] = eval(config.get(section, 'activation'))
        if config.has_option(section, 'layer_norm'):
            network_args['layer_norm'] = config.getboolean(section, 'layer_norm')
        self._network_fn = get_network_builder('mlp')(**network_args)

class LSTM(NetworkFn):

    def __init__(self, config, section):
        network_args = {}
        if config.has_option(section, 'nlstm'):
            network_args['nlstm'] = config.getint(section, 'nlstm')
        if config.has_option(section, 'layer_norm'):
            network_args['layer_norm'] = config.getboolean(section, 'layer_norm')
        self._network_fn = get_network_builder('lstm')(**network_args)


class MLP_LSTM_MLP(NetworkFn):

    def __init__(self, config, section):

        network_args = {}
        if config.has_option(section, 'nlstm'):
            network_args['nlstm'] = config.getint(section, 'nlstm')
        if config.has_option(section, 'layer_norm_lstm'):
            network_args['layer_norm_lstm'] = config.getboolean(section, 'layer_norm_lstm')
        if config.has_option(section, 'num_layers_in'):
            network_args['num_layers_in'] = config.getint(section, 'num_layers_in')
        if config.has_option(section, 'num_hidden_in'):
            network_args['num_hidden_in'] = config.getint(section, 'num_hidden_in')
        if config.has_option(section, 'layer_norm_in'):
            network_args['layer_norm_in'] = config.getboolean(section, 'layer_norm_in')
        if config.has_option(section, 'num_layers_out'):
            network_args['num_layers_out'] = config.getint(section, 'num_layers_out')
        if config.has_option(section, 'num_layers_out'):
            network_args['num_layers_out'] = config.getint(section, 'num_layers_out')
        if config.has_option(section, 'layer_norm_out'):
            network_args['layer_norm_out'] = config.getboolean(section, 'layer_norm_out')
        if config.has_option(section, 'activation'):
            network_args['activation'] = eval(config.get(section, 'activation'))

        def mlp_lstm_mlp(nlstm=128, layer_norm_lstm=False, num_layers_in=0, num_hidden_in=0, num_layers_out=0, \
                         num_hidden_out=0, activation=tf.tanh, layer_norm_in=False, layer_norm_out=False):

            def network_fn(X, nenv=1):
                nbatch = X.shape[0]
                nsteps = nbatch // nenv

                h = X
                with tf.variable_scope('mlp_in', reuse=tf.AUTO_REUSE):
                    for i in range(num_layers_in):
                        h= fc(h, 'mlp_in_fc{}'.format(i), nh=num_hidden_in, init_scale=np.sqrt(2))
                        if layer_norm_in:
                            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
                        h = activation(h)

                h = tf.layers.flatten(X)

                M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
                S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

                xs = batch_to_seq(h, nenv, nsteps)
                ms = batch_to_seq(M, nenv, nsteps)

                if layer_norm_lstm:
                    h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
                else:
                    h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

                h = seq_to_batch(h5)

                with tf.variable_scope('mlp_out', reuse=tf.AUTO_REUSE):
                    for i in range(num_layers_out):
                        h = fc(h, 'mlp_out_fc{}'.format(i), nh=num_hidden_out, init_scale=np.sqrt(2))
                        if layer_norm_out:
                            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
                        h = activation(h)

                initial_state = np.zeros(S.shape.as_list(), dtype=float)

                return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

            return network_fn

        self._network_fn = mlp_lstm_mlp(**network_args)


def get_network(config, section):

    """
    Returns callable class for building network. The section in config can either have 'type' or 'entrypoint' as option
    The type can be set to mlp, lstm, etc,. The entrypoint option specifies the entrypoint to class definition
    that inherits NetworkFn. This however is not a strict requirement as long as the object returns a network
    when called.

    Refer baselines.common.models for the complete list of values type can take and the requirements for using custom
    network function.

    Example:

    [my_network]
    type = mlp
    num_layers = 2
    num_hidden = 128

    or

    [my_network]
    entrypoint: roam_rl.openai_baselines.models:MLP
    num_layers = 2
    num_hidden = 128

    """

    assert (config.has_option(section, 'type') and config.has_option(section, 'entrypoint')) is False,\
    "cannot specify both type and entrypoint"

    _mapping = {
        'mlp': MLP,
        'lstm': LSTM,
        'mlp_lstm_mlp':MLP_LSTM_MLP
    }

    if config.has_option(section, 'type'):
        _type = config.get(section, 'type')
        return _mapping[_type](config, section)
    elif config.has_option(section, 'entrypoint'):
        return initfromconfig(config, section)
    else:
        raise ValueError("network unknown")
