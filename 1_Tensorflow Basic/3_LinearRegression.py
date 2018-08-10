# -*-coding:utf8-*-
# Created by Rachel.minii

import tensorflow as tf

'''
@ 여기서 집중할것!

* name 의 의미 : 살펴보기 쉽게 하기 위함으로, tensor 를 print 할 때도 이름으로 표현된다 
               name 안쓰면 Placeholder 로 나와버림..
'''

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# name : 나중에 텐서보드 등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기위함
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

# X 와 Y 의 상관관계 분석을 위한 linear 식
hypothesis = W * X + b

# loss function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 텐서플로우에 포함되어있는 Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# cost 를 최소화 시키는 것이 목표
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화 100번 수행
    for step in range(100):
        # session.run 을 통해 train_op 와 cost 그래프를 수행한다
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 결과 확인
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))


