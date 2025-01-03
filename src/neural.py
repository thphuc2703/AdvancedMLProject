from tensorflow.keras.layers import BatchNormalization
import math
def bn(input, is_training_phase, bn_scope):
    bn_layer = BatchNormalization(name=bn_scope)
    return bn_layer(input, training=is_training_phase)

def tanh(input, bn_scope=None, is_training_phase=None, dropout_value=None, name="relu"):
    if bn_scope:
        input = bn(input, is_training_phase, bn_scope)

    value = tf.tanh(input)

    if dropout_value is not None:
        return tf.nn.dropout(value, rate=dropout_value, name=name)  # Use 'rate' instead of 'keep_prob'
    else:
        return value

def softmax(input, bn_scope=None, is_training_phase=None):
    if bn_scope and is_training_phase is not None:
        input = bn(input, is_training_phase, bn_scope)
    return tf.nn.softmax(input)
def get_z(input, batch_size, z_size, W_mean, W_stddev, b_mean, b_stddev, is_prior):
    mean = tf.tensordot(input, W_mean, axes=1) + b_mean
    stddev = tf.tensordot(input, W_stddev, axes=1) + b_stddev
    stddev = tf.sqrt(tf.exp(stddev))
    
    epsilon = tf.random.normal(shape=[batch_size, z_size], name='epsilon')

    z = mean if is_prior else mean + tf.multiply(stddev, epsilon)

    pdf_z = tfp.distributions.Normal(loc=mean, scale=stddev)

    return z, pdf_z

def iter(size, func, iter_arg, iter_arg2=None, iter_arg3=None, **kwargs):
    ta = tf.TensorArray(tf.float32, size=size)
    loop_init = (0, ta)

    def cond(i, _):
        return i < size

    def body(i, ta):
        if iter_arg3 is not None:
            return i + 1, ta.write(i, func(iter_arg[i], iter_arg2[i], iter_arg3[i], **kwargs))
        else:
            return i + 1, ta.write(i, func(labels=iter_arg[:, i], logits=iter_arg2[:, i]))

    _, ta_final = tf.while_loop(cond, body, loop_init)

    return ta_final.stack()
def get_r(max_len_st, alpha, with_lt=None):
    """
        r for [0, max_len_st]
    """
    r = lambda t: math.exp(alpha * (t - max_len_st))
    seq_len = max_len_st + 1 if with_lt else max_len_st
    r_arr = [r(t) for t in range(seq_len)]
    r_vec = tf.constant(r_arr, shape=(seq_len, 1), dtype=tf.float32)
    return r_vec
