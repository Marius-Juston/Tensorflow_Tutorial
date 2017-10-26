import tensorflow as tf

from Helper_Methods import open_tensorboard

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# We've created a model, but we don't know how good it is yet. To evaluate the model on training data, we need a y
# placeholder to provide the desired values, and we need to write a loss function.

# A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model
#  for linear regression, which sums the squares of the deltas between the current model and the provided data (r^2).
# linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square
# to square that error. Then, we sum all the squared errors to create a single scalar that abstracts the error of all
#  Examples using tf.reduce_sum:

# Values expected from tensor
y = tf.placeholder(tf.float32)

# subtracts every returned value from the expected value given the x value and then squares the error
squared_deltas = tf.square(linear_model - y)

# Sums all the errors to only a single number
loss = tf.reduce_sum(squared_deltas)

EXPECTED_LABELS = [0, -1, -2, -3]

loss_value, y_expected = sess.run([loss, linear_model], {x: [1, 2, 3, 4], y: EXPECTED_LABELS})

print("Expected y values y:", EXPECTED_LABELS)
print("Returned y values:", y_expected)
print("R2 Loss:", loss_value)
# producing the loss value
# 23.66

# We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1. A variable
# is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign. For
# example, W=-1 and b=1 are the optimal parameters for our model. We can change W and b accordingly:

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print()
print("Output with optimized weights and biases:")
loss_value, y_expected = sess.run([loss, linear_model], {x: [1, 2, 3, 4], y: EXPECTED_LABELS})
print("Expected y values y:", EXPECTED_LABELS)
print("Returned y values:", y_expected)
print("R2 Loss:", loss_value)

# The final print shows the loss now is zero. 0.0 We guessed the "perfect" values of W and b, but the whole point of
# machine learning is to find the correct model parameters automatically. We will show how to accomplish this in the
# next section.

open_tensorboard(__file__, sess)
