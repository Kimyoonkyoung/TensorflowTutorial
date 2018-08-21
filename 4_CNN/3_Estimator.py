# -*-coding:utf8-*-
# Created by Rachel.minii
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# model param
model_params = {"learning_rate": 1e-4, "dropout": 0.5}


# generate model_fn
def cnn_model_fn(features, labels, mode, params):
    # input layer
    x_image = tf.reshape(features['x'], [-1, 28, 28, 1])  # (N, H, W, C)

    # conv layer 1
    conv1 = tf.layers.conv2d(x_image, 32, [5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.constant_initializer(0.1))
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # conv layer 2
    conv2 = tf.layers.conv2d(pool1, 64, [5, 5],
                             activation=tf.nn.relu,
                             padding='same',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.constant_initializer(0.1))
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # fully-connected layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(pool2_flat, 1024,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            bias_initializer=tf.constant_initializer(0.1))
    dropout = tf.layers.dropout(dense,
                                rate=params['dropout'],
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(dropout, 10, activation=None)

    # prediction
    predictions = {
        "classes": tf.argmax(input=logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss (for both TRAIN, EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# MNIST 데이터 불러오기 위한 함수 정의
def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.astype('float32') / 255.
    train_y = train_y.astype('int32')

    test_x = test_x.astype('float32') / 255.
    test_y = test_y.astype('float32') / 255.

    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_load()


'''
# 본격적으로 estimator 를 만듦
'''
# Generate Estimator
CNN = tf.estimator.Estimator(model_fn=cnn_model_fn,
                             params=model_params,
                             model_dir='./model/mnist_cnn')  # 모델 저장하는 경로

# train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_x},
    y=train_y,
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
CNN.train(input_fn=train_input_fn, steps=1000)

# eval the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': test_x},
    y=test_y,
    num_epochs=1,
    shuffle=False
)
eval_results = CNN.evaluate(input_fn=eval_input_fn)
print(eval_results)




