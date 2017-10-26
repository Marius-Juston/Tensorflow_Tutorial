import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b
# linear_model = tf.add(tf.mul(W , x) , b)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# However, TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
# The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative
# of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and
# error-prone. Consequently, TensorFlow can automatically produce derivatives given only a description of the model
# using the function tf.gradients. For simplicity, optimizers typically do this for you. For example,


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  # reset values to incorrect defaults.
for i in range(1000000):
    t = sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# Prints out the found "perfect" values for the weight and bias given the training data
print(sess.run([W, b]))
