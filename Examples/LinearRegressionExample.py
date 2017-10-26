# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from Helper_Methods import open_tensorboard

data = make_regression(noise=10, n_informative=1, n_features=1, random_state=42)

features = MinMaxScaler().fit_transform(data[0].reshape(-1, 1))
labels = MinMaxScaler().fit_transform(data[1].reshape(-1, 1))

plt.scatter(features, labels)

min_x, max_x = min(features), max(features)

tf.set_random_seed(42)

# weight = tf.Variable(tf.truncated_normal([1]), dtype=tf.float32,name="weight")
# tf.summary.histogram("weight",weight)

weight = tf.Variable(0, dtype=tf.float32, name="weight")
tf.summary.scalar("weight", weight)

# bias = tf.Variable(tf.truncated_normal([1]), dtype=tf.float32, name="bias")
# tf.summary.histogram("bias", bias)

bias = tf.Variable(1, dtype=tf.float32, name="bias")
tf.summary.scalar("bias", bias)

x = tf.placeholder(tf.float32)
true_y = tf.placeholder(tf.float32)

linear_model = tf.multiply(weight, x) + bias

loss = tf.reduce_sum(tf.square(linear_model - true_y))
tf.summary.scalar("loss", loss)

train = tf.train.GradientDescentOptimizer(0.001, name="optimizer").minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

file_writer, _ = open_tensorboard(__file__, sess)

merge = tf.summary.merge_all()

plt.ion()
plt.pause(0.0001)

min_y, max_y = sess.run(linear_model, {x: [min_x, max_x]})
line = plt.plot([min_x, max_x], [min_y, max_y], c="red")[0]
plt.pause(0.0001)

for i in range(1000):
    _, l, summary, w, b = sess.run([train, loss, merge, weight, bias], {x: features, true_y: labels})
    min_y, max_y = sess.run(linear_model, {x: [min_x, max_x]})

    line.set_ydata([min_y, max_y])

    print("Iteration:", i, " Loss:", l, "Weight:", w, "Bias:", b)
    file_writer.add_summary(summary, i)
    plt.pause(0.0001)

plt.show(block=True)
