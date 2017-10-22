# The completed trainable linear regression model is shown here:

import tensorflow as tf

# Model parameters
from Helper_Methods import open_tensorboard

W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")

# Model input and output
x = tf.placeholder(tf.float32, name="x")

linear_model = W * x + b
y = tf.placeholder(tf.float32, name="y")

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01, name="gradients")
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

# Visualization --------------------------
tf.summary.scalar("loss", loss)
tf.summary.histogram("weight", W)
tf.summary.histogram("bias", b)

merge = tf.summary.merge_all()

file_writer, _ = open_tensorboard(__file__, sess)
#  ---------------------------------------

for i in range(1000):
    _, summary = sess.run([train, merge], {x: x_train, y: y_train})
    file_writer.add_summary(summary, i)

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
