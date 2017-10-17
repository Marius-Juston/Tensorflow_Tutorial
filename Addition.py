import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()

node3 = tf.add(node1, node2)

# node3 = node1 + node2

print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# The last two print statements produce

# node3: Tensor("Add:0", shape=(), dtype=float32)
# sess.run(node3): 7.0
