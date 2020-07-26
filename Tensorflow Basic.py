import tensorflow as tf
import numpy as np
print(tf.__version__)

tf.compat.v1.disable_eager_execution()


tf.compat.v1.reset_default_graph()
a = tf.compat.v1.placeholder(np.float32, (2, 2))
b = tf.compat.v1.Variable(tf.ones((2, 2)))
c = a @ b

print(c)

s = tf.compat.v1.InteractiveSession()


s.run(tf.compat.v1.global_variables_initializer())
s.run(c, feed_dict={a: np.ones((2, 2))})


s.close()


tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32, trainable=True)
f = x ** 2


optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f, var_list=[x])



tf.compat.v1.trainable_variables()


with tf.compat.v1.Session() as s:  # in this way session will be closed automatically
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        _, curr_x, curr_f = s.run([step, x, f])
        print(curr_x, curr_f)


# ### Simple optimization (with tf.Print)

tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32)
f = x ** 2
f = tf.compat.v1.Print(f, [x, f], "x, f:")

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)


with tf.compat.v1.Session() as s:
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        s.run([step, f])


# Prints to jupyter server stdout (not available in Coursera Notebooks):
# 2018-07-21 18:01:27.308270: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-1.0670249][1.1385423]
# 2018-07-21 18:01:27.308809: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.85361993][0.72866696]
# 2018-07-21 18:01:27.309116: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.68289596][0.46634689]
# 2018-07-21 18:01:27.309388: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.54631674][0.29846197]
# 2018-07-21 18:01:27.309678: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.43705338][0.19101566]
# 2018-07-21 18:01:27.309889: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.34964269][0.12225001]
# 2018-07-21 18:01:27.310213: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.27971417][0.078240015]
# 2018-07-21 18:01:27.310475: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.22377133][0.050073609]
# 2018-07-21 18:01:27.310751: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.17901707][0.032047112]
# 2018-07-21 18:01:27.310963: I tensorflow/core/kernels/logging_ops.cc:79] x, f:[-0.14321366][0.020510152]


# ### Simple optimization (with TensorBoard logging)

tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32)
f = x ** 2

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)


tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)



s = tf.compat.v1.InteractiveSession()
summary_writer = tf.compat.v1.summary.FileWriter("logs/1", s.graph)
s.run(tf.compat.v1.global_variables_initializer())

summaries = tf.compat.v1.summary.merge_all()

for i in range(1, 10):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()


# Run  `tensorboard --logdir=./logs` in bash

import os
os.system("tensorboard --logdir=./logs --host 0.0.0.0 --port 6006 &")

s.close()


# ### Training a linear model
# generate model data
N = 1000
D = 3
x = np.random.random((N, D))
w = np.random.random((D, 1))
y = x @ w + np.random.randn(N, 1) * 0.20

print(x.shape, y.shape)
print(w.T)


# In[22]:


tf.compat.v1.reset_default_graph()

features = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
target = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

weights = tf.compat.v1.get_variable("weights", shape=(D, 1), dtype=tf.float32)
predictions = features @ weights

loss = tf.reduce_mean((target - predictions) ** 2)

print(target.shape, predictions.shape, loss.shape)


optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)


with tf.compat.v1.Session() as s:
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(300):
        _, curr_loss, curr_weights = s.run([step, loss, weights],
                                           feed_dict={features: x, target: y})
        if i % 50 == 0:
            print(curr_loss)


# found weights
print(curr_weights.T)

# true weights
print(w.T)