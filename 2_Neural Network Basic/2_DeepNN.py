# -*-coding:utf8-*-
# Created by Rachel.minii

# 털과 날개가 있는지 없는지에 따라 포유류인지 조류인지 분류하는 신경망 모델

import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

'''
# 신경망 모델 구성
'''
X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')

# [input, hidden]
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
b1 = tf.Variable(tf.zeros([10]))

# [hidden, output]
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))
b2 = tf.Variable(tf.zeros([3]))

# 적용
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 최종 아웃풋
model = tf.add(tf.matmul(L1, W2), b2)

# 텐서플로우 내의 cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

'''
# 신경망 모델 학습
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


'''
# 결과 확인
'''
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
