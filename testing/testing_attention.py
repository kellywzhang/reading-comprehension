import tensorflow as tf
import numpy as np

attending_size = 5
attended_size= 4
batch_size = 2
# time = 3

# MUST GET THESE PARAMS FROM DATA IN PRE-PROCESSING STAGE
max_time = 10
max_entities = 6

# Starting interactive Session
sess = tf.InteractiveSession()

# 3, 2, 4
# Expect dimensions: attending (batch x attending_size), attended (batch x time x attended_size)
attending = tf.constant([[0,1,2,3,4],[4,3,2,1,0]], dtype=tf.float32)
attended = tf.constant([[[0,1,2,4],[0,1,1,5],[0,1,2,2]],[[0,1,2,7],[0,1,0,2],[0,1,2,0]]], dtype=tf.float32) #2,4,3

W_bilinear = tf.get_variable(name="bilinear_attention", shape=[attending_size, attended_size], \
    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

# Dimensions (batch x attended_size)
attending_tensor = tf.matmul(attending, W_bilinear)
attending_tensor = tf.reshape(attending_tensor, [batch_size, attended_size, 1])

# multiplies each slice with each other respective slice - EXPLAIN BETTER
dot_prod = tf.batch_matmul(attended, attending_tensor)
# Should return matrix of attention weights with dimensions (batch x time)
dot_prod = tf.squeeze(dot_prod)

num = tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_sum(dot_prod, 1), -1))) #batch x time
#tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_max(dot_prod, 1), -1)))
denom = tf.reduce_sum(num, 1) # batch x ,

# get 1/denom so can multiply with numerator
#inv = tf.truediv(tf.ones(denom.get_shape()), denom)


#result = tf.transpose(tf.mul(tf.transpose(num), inv))
#result = tf.transpose(tf.div(tf.transpose(num), denom))
weights = tf.div(num, tf.expand_dims(denom, 1))
# batch x time x
weighted = tf.mul(attended, tf.expand_dims(weights, -1))
# batch by attended size
result = tf.reduce_sum(weighted, 1)

"""
# Find weighted sum of attended tensor using alpha_weights
# attended dimensions: (batch x time x attended_size)
# result (batch x time)
attended = tf.transpose(attended, perm=[2,0,1])
result = tf.mul(attended, result)
result = tf.transpose(result, perm=[1,2,0])
result = tf.reduce_sum(result, 1)
"""

#result = tf.nn.softmax(dot_prod)

sess.run(tf.initialize_all_variables())

print("attended")
print(attended.eval())
print(attended.get_shape())
#print(attended.get_shape())
#print(result.get_shape())
print("attending")
print(attending_tensor.eval())
print(attending.get_shape())
print("dot prod")
print(dot_prod.eval())
print(num.eval())
print(num.get_shape())
print(denom.eval())
print(denom.get_shape())
print("result")
print(weights.eval())
print(weights.get_shape())
print("wegithed")
print(weighted.eval())
print('result')
print(result.eval())
print(result.get_shape())

#print(attending_tensor.eval())
#print(denom.eval())
#print(denom.get_shape())
#print(denom_prime.eval())
#print(result.eval())

#print(num.get_shape())

sess.close()
