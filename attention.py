"""
Goal: Create classes to easily implement different attention mechanisms.

Issues:
    - MUST MAKE SOFTMAX NUMERICALLY STABLE
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
    def __call__(self, attending, attended, time_mask, scope=None):
      with tf.variable_scope(self._scope or type(self).__name__):  # "BilinearFunction"
          attending_size = self._attending_size
          attended_size= self._attended_size

          batch_size = tf.shape(attended)[0]

          # different initializer?
          self.W_bilinear = tf.get_variable(name="bilinear_attention", shape=[attending_size, attended_size], \
              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

          # Dimensions (batch x attended_size)
          attending_tensor = tf.matmul(attending, self.W_bilinear)
          attending_tensor = tf.reshape(attending_tensor, [batch_size, attended_size, 1])
          #self.attending_tensor = attending_tensor

          #self.attended = attended

          # Now take dot products of attending tensor with each timestep of attended tensor
          # Should return matrix of attention weights with dimensions (batch x time)

          # multiplies each slice with each other respective slice - EXPLAIN BETTER
          dot_prod = tf.squeeze(tf.batch_matmul(attended, attending_tensor)) * time_mask
          #self.dot_prod = dot_prod

	      # Should return matrix of attention weights with dimensions (batch x time)
          #dot_prod = tf.squeeze(dot_prod)

          # Custom Softmax b/c need to use time_mask --------------------
          # Also numerical stability: alpha_weights = tf.nn.softmax(dot_prod)

          # e_x = exp(x - x.max(axis=1))
          # out = e_x / e_x.sum(axis=1)
          numerator = tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_max(dot_prod, 1), -1))) * time_mask
          denom = tf.reduce_sum(numerator, 1)
          #self.numerator = numerator
          #self.denom = denom

          # Dimensions (batch x time)
          alpha_weights = tf.div(numerator, tf.expand_dims(denom, 1))

          # Find weighted sum of attended tensor using alpha_weights
          # attended dimensions: (batch x time x attended_size)
          attended_weighted = tf.mul(attended, tf.expand_dims(alpha_weights, -1))
          #self.attended_weighted = attended_weighted

          # attend_result dimensions (batch x attended_size)
          attend_result = tf.reduce_sum(attended_weighted, 1)

          return (alpha_weights, attend_result)
