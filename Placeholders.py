# A graph can be parametrized to accept external inputs, known as placeholders. A placeholder is a promise to
# provide a value later.

import tensorflow as tf

# placeholder are different from variable as you assign the value to the placeholder manually in sess.run() while
# variables are assigned directly in the contractor
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b)
# and then an operation on them

sess = tf.Session()

# Feed dict are dictionary where the key (thing before the colon) is the variable you wish to assign a value to and
# the value (on the right of the colon) is the value you want to assign the value of the variable

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Result
# 7.5
# [ 3.  7.]
