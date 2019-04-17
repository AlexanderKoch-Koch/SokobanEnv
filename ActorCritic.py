import numpy as np
import tensorflow as tf
from sokoban_env import SokobanEnv
from Models import simple_DNN
from tensorboard_logger import configure, log_value
import random
import time

env = SokobanEnv()
configure("runs/AC_9_fast")
gamma = 0.95
epsilon = 0.8
tf_input = tf.placeholder(dtype=tf.float32, shape=(100,))
tf_return = tf.placeholder(dtype=tf.float32)
tf_action = tf.placeholder(dtype=tf.int32)

l0 = tf.transpose(tf.expand_dims(tf_input, axis=1))
tf_w1 = tf.Variable(tf.truncated_normal(shape=(100, 200)))
tf_b1 = tf.Variable(tf.truncated_normal(shape=(200,)))
tf_l1 = tf.nn.relu(tf.add(tf.matmul(l0, tf_w1), tf_b1))

tf_w2 = tf.Variable(tf.truncated_normal(shape=(200, 100)))
tf_b2 = tf.Variable(tf.truncated_normal(shape=(100,)))
tf_l2 = tf.nn.relu(tf.add(tf.matmul(tf_l1, tf_w2), tf_b2))

# Actor head
tf_actor_w3 = tf.Variable(tf.truncated_normal(shape=(100, 4)))
tf_actor_b3 = tf.Variable(tf.truncated_normal(shape=(4,)))
tf_actor_output = tf.add(tf.matmul(tf_l2, tf_actor_w3), tf_actor_b3)

tf_log_output = tf.math.multiply(tf.log(tf_actor_output), tf.one_hot(tf_action, on_value=tf_return, depth=4))
loss = -tf.reduce_sum(tf_log_output)
tf_actor_optimizer = tf.train.AdamOptimizer().minimize(loss)

# Critic head
tf_critic_w3 = tf.Variable(tf.truncated_normal(shape=(100, 1)))
tf_critic_b3 = tf.Variable(tf.truncated_normal(shape=(1,)))
tf_critic_output = tf.add(tf.matmul(tf_l2, tf_critic_w3), tf_critic_b3)[0][0]

tf_critic_y = tf.placeholder(dtype=tf.float32)
tf_critic_loss = tf.square(tf.subtract(tf_critic_output, tf_critic_y))
tf_critic_optimizer = tf.train.AdamOptimizer().minimize(tf_critic_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(10000):
        # env.render()
        epsilon *= 0.9998
        state = env.reset().flatten()
        done = False
        state_value = sess.run(tf_critic_output, feed_dict={tf_input: state})
        reward_sum = 0
        state_value_sum = state_value
        step = 0
        while not done:
            start = time.time()
            actions = sess.run(tf_actor_output, feed_dict={tf_input: state})
            print("DNN took " + str(time.time() - start) + "s")
            if random.randint(0, 100) > epsilon * 100:
                action = np.argmax(actions[0])
            else:
                action = random.randint(0, 3)

            start = time.time()
            next_state, reward, done, _ = env.step(action)
            print("step took " + str(time.time() - start) + "s")
            if done:
                break

            next_state = next_state.flatten()
            step += 1
            reward_sum += reward
            value_next_state = sess.run(tf_critic_output, feed_dict={tf_input: next_state})
            state_value_sum += value_next_state
            td_error = reward + gamma * value_next_state - state_value

            sess.run(tf_critic_optimizer, feed_dict={tf_input: state, tf_critic_y: reward + gamma * value_next_state})
            sess.run(tf_actor_optimizer, feed_dict={tf_input: state, tf_return: td_error, tf_action: action})

            state = next_state
            state_value = value_next_state

        log_value("reward_sum", reward_sum, episode)
        log_value("epsilon", epsilon, episode)
        log_value("avg_value_prediction", state_value_sum / step, episode)
