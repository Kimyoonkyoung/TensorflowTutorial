# -*-coding:utf8-*-
# Created by Rachel.minii

import tensorflow as tf

'''
@ 여기서 집중할것!
placeholder 와 variable 의 차이

* placeholder 는 "입력"
* variable 은 "계산하면서 변경될 최적화 대상 변수"
'''

# tf.placeholder : 계산을 '실행' 할때 '입력'값을 받는 변수
# None 은 크기가 정해지지 않았음을 의미
X = tf.placeholder(tf.float32, [None, 3])
print(X)

# X 에 넣을 값, 2번째 차원은 3개로
x_data = [[1,2,3], [4,5,6]]

# tf.Variable : 그래프를 계산하면서 최적화할 변수
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

# 입력한 변수와 계산할 수식
expr = tf.matmul(X, W) + b

sess = tf.Session()

# 모든 변수가 아닌 Variable 변수를 초기화하기위한 작업
init = tf.global_variables_initializer()
sess.run(init)

print("\n=== x_data ===")
print(x_data)
print("\n=== W ===")
print(sess.run(W))
print("\n=== b ===")
print(sess.run(b))
print("\n=== expr ===")

# 여기서 W, b 는 Variable 이었고, X 가 placeholder 이었으니, X 를 초기화해주어야한다
print(sess.run(expr, feed_dict={X : x_data}))

sess.close()
