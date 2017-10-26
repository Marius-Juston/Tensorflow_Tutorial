# ------------------ CODE FROM Constants.py ------------------#
import tensorflow as tf

from Helper_Methods import open_tensorboard

node1 = tf.constant(3.0, dtype=tf.float32, name="node1")
node2 = tf.constant(4.0, name="node1")

sess = tf.Session()

# ------------------------------------------------------------#

# You can create more complicated computation by combining Tensor nodes with operations (Operations are also nodes)
# Examples:
# Add
addition_node = tf.add(node1, node2)

# Can also use the + to add two nodes together
# addition_node = node1 + node2

print("Addition")
print("add_node:", addition_node)
print("sess.run(add_node):", sess.run(addition_node))

# Subtract
subtract_node = tf.subtract(node1, node2)

# Can also use the - to subtract two nodes together
# subtract_node = node1 - node2

print()
print("Subtraction")
print("subtract_node:", subtract_node)
print("sess.run(subtract_node):", sess.run(subtract_node))

# Multiply
multiplication_node = tf.multiply(node1, node2)

# Can also use the * to multiply two nodes together
# multiplication_node = node1 * node2

print()
print("Multiplication")
print("multiplication_node:", multiplication_node)
print("sess.run(multiplication_node):", sess.run(multiplication_node))

# 4) Divide
division_node = tf.divide(node1, node2)

# Can also use the / to divide two nodes together
# division_node = node1 / node2

print()
print("Division")
print("division_node:", division_node)
print("sess.run(division_node):", sess.run(division_node))

file_writer, path = open_tensorboard(__file__, sess)
