# -*-coding:utf8-*-
# Created by Rachel.minii
import tensorflow as tf

'''
@ 여기서 집중할것!

* session 을 열어주고, run 을 해야한다는것
* 모든것들은 tensor 라는 것
* session.run() 이 꼭 어떤 operation 이 아니고 변수 값을 확인하기 위함으로도 쓸 수 있다는것
'''

# tf.constant : 상수
hello = tf.constant('Hello, tensorflow')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
d = a + b
print(c)
print(d)

# session 을 정의해줘야 진짜 실행
sess = tf.Session()

# sess.run : 설정한 텐서그래프를 실행
print(sess.run(hello))
print(sess.run([a,b,c,d]))

# sess 닫기
sess.close()


'''
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
이 에러를 만날 수 있음
-> 바이너리를 통한 텐서플로우를 설치했을 때, AVX2 와 FMA 를 지원하지 않아서 성능저하가 있을 수 있다는 안내문이다
-> 직접 빌드해서 사용하면 안내문이 안뜨게 할 수 있다
-> http://www.kwangsiklee.com/2017/04/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EA%B2%BD%EA%B3%A0%EB%A9%94%EC%84%B8%EC%A7%80-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-the-tensorflow-library-wasnt-compiled-to-use-sse3-instructions/
'''