# -*-coding:utf8-*-
# Created by Rachel.minii
import tensorflow as tf
import numpy as np

'''
@ 여기서 집중할것!

* Saver !!
* global step : 내가 counting 하지 않아도 알아서 step 을 알 수 있다
'''

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
# 학습에 참여시키지않고 학습횟수에 따라 단순히 증가시킬 변수
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

# global step 으로 넘겨준변수를, 학습용 변수를 최적화 할 때 마다 학습횟수는 하나씩 증가시킴
train_op = optimizer.minimize(cost, global_step=global_step)


'''
# 신경망 모델 학습
'''
sess = tf.Session()

# 모델을 저장하고 불러오는 API를 초기화
# global_variables 함수를 통해 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# 최적화 진행
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))


# 최적화 후 변수 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)


'''
결과확인
'''
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

