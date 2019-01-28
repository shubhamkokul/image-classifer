import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4, 2], name="x-input")

t_ = tf.placeholder(tf.float32, shape=[4, 1], name="t-input")
w11_22 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="w11_22")
w31_32 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="w31_32")

print(w11_22)
print(w31_32)
Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")
Layer1 = tf.sigmoid(tf.matmul(x_, w11_22) + Bias1)
y_ = tf.sigmoid(tf.matmul(Layer1, w31_32) + Bias2)
cost = tf.reduce_mean(tf.square(y_ - t_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
x_value = [[0, 0], [0, 1], [1, 0], [1, 1]]
t_value = [[0], [1], [1], [0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10):
    sess.run(train_step, feed_dict={x_: x_value, t_: t_value})
    if i % 10000 == 0:
        print('Epoch ', i)
        print('x_', x_value)
        print('y_ ', sess.run(y_, feed_dict={x_: x_value, t_: t_value}))
        print('w11_22 ', sess.run(w11_22))
        print('Bias1 ', sess.run(Bias1))
        print('w31_32 ', sess.run(w31_32))
        print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: x_value, t_: t_value}))