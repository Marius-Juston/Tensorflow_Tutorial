# The canonical import statement for TensorFlow programs is as follows:
# This gives Python access to all of TensorFlow's classes, methods, and symbols.
import tensorflow as tf

from Helper_Methods import open_tensorboard

# One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores
#  internally

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# The final print statement produces
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# It prints this instead of the expected 3.0 and 4.0 because all the nodes (constant, Variable, placeholder, etc) are
# evaluated during the session run time and so you have to ask the current session to evaluate the tensors

# The following code creates a Session object and then invokes its run method to run enough of the computational
# graph to evaluate node1 and node2
sess = tf.Session()
print(sess.run([node1, node2]))

# we see the expected values of 3.0 and 4.0:
# [3.0, 4.0]



open_tensorboard(__file__, sess)
