"""
Goal: Create classes to easily implement different attention mechanisms.

Issues:
    - MUST MAKE SOFTMAX NUMERICALLY STABLE
    - PERHAPS SHOULD RETURN WEIGHTS AS WELL AS WEIGHTED SUM OF ATTENDED
    - Find out what initializer to user for bilinear weight

Credits: Idea from https://arxiv.org/pdf/1606.02858v2.pdf
"""

import tensorflow as tf

class BilinearFunction(object):
    def __init__(self, attending_size, attended_size, scope=None):
      self._attending_size = attending_size
      self._attended_size = attended_size
      self._scope = scope

    # Expect dimensions: attending (batch x attending_size),
        # attended (batch x time x attended_size) - time could be other dim value
    def __call__(self, attending, attended, seq_lens, scope=None):
      with tf.variable_scope(self._scope or type(self).__name__):  # "BilinearFunction"
          attending_size = self._attending_size
          attended_size= self._attended_size

          # different initializer?
          W_bilinear = tf.get_variable(name="bilinear_attention", shape=[attending_size, attended_size], \
              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

          # Dimensions (batch x attended_size)
          attending_tensor = tf.matmul(attending, W_bilinear)
          attending_tensor = tf.reshape(attending_tensor, [batch_size, attended_size, 1])

          # Now take dot products of attending tensor with each timestep of attended tensor
          # Should return matrix of attention weights with dimensions (batch x time)

          # multiplies each slice with each other respective slice - EXPLAIN BETTER
          dot_prod = tf.batch_matmul(attended, attending_tensor)
          # Should return matrix of attention weights with dimensions (batch x time)
          dot_prod = tf.squeeze(dot_prod)

          # Dimensions (batch x time)
          seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)

          tf.exp(dot_prod * seq_len_mask)
          tf.reduce_sum(tf.exp(tf.exp(dot_prod * seq_len_mask)), 1)

          # Custom Softmax b/c need to use time_mask
          # Also numerical stability: alpha_weights = tf.nn.softmax(dot_prod)

          numerator = tf.exp(dot_prod * seq_len_mask) #batch x time
          denom = tf.reduce_sum(tf.exp(dot_prod * seq_len_mask), 1)

          # get 1/denom so can multiply with numerator
          inv = tf.truediv(tf.ones(denom.get_shape()), denom)
          # Transpose so broadcasting scalar multiplication works properly
          # Dimensions (batch x time)
          alpha_weights = tf.transpose(tf.mul(tf.transpose(numerator), inv))

          # Find weighted sum of attended tensor using alpha_weights
          # attended dimensions: (batch x time x attended_size)
          tf.mul(attended, alpha_weights)

          # Again must permute axes so broadcasting scalar multiplication works properly
          attended_transposed = tf.transpose(attended, perm=[2,0,1])
          attended_weighted_transposed = tf.mul(attended_transposed, alpha_weights)
          attended_weighted = tf.transpose(attended_weighted_transposed, perm=[1,2,0])
          # attend_result dimensions (batch x attended_size)
          attend_result = tf.reduce_sum(attended_weighted, 1)

          return (alpha_weights, attend_result)
