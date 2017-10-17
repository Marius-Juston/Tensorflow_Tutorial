# The canonical import statement for TensorFlow programs is as follows:

import tensorflow as tf

# This gives Python access to all of TensorFlow's classes, methods, and symbols. Most of the documentation assumes
# you have already done this. One type of node is a constant. Like all TensorFlow constants, it takes no inputs,
# and it outputs a value it stores internally
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# The final print statement produces
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run([node1, node2]))

# we see the expected values of 3.0 and 4.0:
# [3.0, 4.0]
