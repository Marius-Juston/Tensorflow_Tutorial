import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

sess = tf.Session()

# We can make the computational graph more complex by adding another operation. For example,
add_and_triple = adder_node * 3.

# The above statement is equivalent to (a + b) * 3 lambda statement

print(sess.run(add_and_triple, {a: 3, b: 4.5}))
# produces the output
# 22.5
