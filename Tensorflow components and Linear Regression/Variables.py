# In machine learning we will typically want a model that can take arbitrary inputs, such as the one above. To make
# the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow
#  us to add trainable parameters to a graph. They are constructed with a type and initial value:

import tensorflow as tf

from Helper_Methods import open_tensorboard

W = tf.Variable(.3, dtype=tf.float32, name="weight")
b = tf.Variable(-.3, dtype=tf.float32, name="biase")
x = tf.placeholder(tf.float32)

linear_model = W * x + b

sess = tf.Session()

# Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are
# not initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program,
# you must explicitly call a special operation as follows:

init = tf.global_variables_initializer()
sess.run(init)

# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# to produce the output
# [ 0.          0.30000001  0.60000002  0.90000004]

open_tensorboard(__file__, sess)
