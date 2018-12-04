# Import required modules
import numpy as np
import tensorflow as tf
import torch
import gym
import matplotlib.pyplot as plt
import argparse
import os

from gym.spaces import Discrete, Box
from tf_utils import *
from spg_tf import *
from spg_torch import *

E = '[ERROR]'
I = '[INFO]'

TF = 'tensorflow'
PT = 'pytorch'

def train_one_epoch(sess):
    # Declaring variables to store epoch details
    batch_acts = []
    batch_len = []
    batch_weights = []
    batch_rews = []
    batch_obs = []

    # Reset env
    obs = env.reset()
    done = False
    ep_rews = []
    rendered_once_in_epoch = False

    while True:

        if not rendered_once_in_epoch:
            env.render()

        batch_obs.append(obs)

        act = sess.run([actions], feed_dict={obs_ph: obs.reshape(1 ,-1)})[0][0]

        # Take the action
        obs, rewards, done, info = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rewards)

        if done:
            # Record info, as episode is complete
            ep_ret = sum(ep_rews)
            ep_len = len(ep_rews)

            batch_rews.append(ep_ret)
            batch_len.append(ep_len)

            batch_weights += [ep_ret] * ep_len

            # Reset the environment
            obs, done, ep_rews = env.reset(), False, []

            rendered_once_in_epoch = True

            if batch_size < len(batch_obs):
                break

    batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              weights_ph: np.array(batch_weights)})

    return batch_loss, batch_rews, batch_len



if '__main__' == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train',
            type=bool,
            default=False,
            help='Set to true if you want to train the model. \
                Default: False')

    parser.add_argument('-g', '--graph',
            type=str,
            default='./graphs/CartPole-v0_graph.pb',
            help='Path to the graph file')

    parser.add_argument('-il', '--input-layer',
            type=str,
            default='input',
            help='The name of the input layer',)

    parser.add_argument('-ol', '--output-layer',
            type=str,
            default='output',
            help='The name of the output layer',)

    parser.add_argument('-e', '--epochs',
            type=int,
            default=50,
            help='The number of epochs')

    parser.add_argument('-gp', '--graph-path',
            type=str,
            default='./graphs/',
            help='Path where the .pb file is saved!')

    parser.add_argument('-f', '--framework',
            type=str,
            default='tensorflow',
            help='Framework to be used - TensorFlow or PyTorch')

    FLAGS, unparsed = parser.parse_known_args()

    # Arguments
    env_name = 'CartPole-v0'
    render = True

    # Create the env
    env = gym.make('CartPole-v0')

    # Get the action space size and observation space size
    act_size = env.action_space.n
    obs_size = env.observation_space.shape[0]

    print ('Action Space Size: {}'.format(act_size),
           '\nObservation Space Size: {}'.format(obs_size))

    #  Choose the framework
    f = FLAGS.framework
    if f != TF and f != PT:
        raise Exception('{}The value of framework can be either \
            tensorflow as pytorch'.format(E))

    if not FLAGS.train:
        if not os.path.exists(FLAGS.graph):
            raise Exception('{}Path to the Graph file does not exists!'.format(E))

        if f == TF:
            test_with_tf(FLAGS)
        elif f == PT:
            test_with_torch(FLAGS)

    else:
        if f == TF:
            train_with_tf(FLAGS)
        elif f == PT:
            train_with_torch(FLAGS, obs_size, act_size)
