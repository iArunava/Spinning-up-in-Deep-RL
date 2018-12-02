
# coding: utf-8

# # Simple Policy Gradient
# 
# This notebook contains the implementation of the Simple Policy Gradient Algorithm using TensorFlow.
# <br/>
# This notebook is created while going through the official Spinning up in Deep RL Docs.

# In[1]:


# Required modules
'''
get_ipython().system('pip install gym')
get_ipython().system('apt-get install python-opengl')
'''


# In[1]:


# Import required modules
import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#from IPython import display


# In[2]:


# Arguments
env_name = 'CartPole-v0'
render = True


# In[3]:


# Create the env
env = gym.make('CartPole-v0')


# In[4]:


# Get the action space size and observation space size
act_size = env.action_space.n
obs_size = env.observation_space.shape[0]

print ('Action Space Size: {}'.format(act_size),
       '\nObservation Space Size: {}'.format(obs_size))


# In[5]:


# Network Hyperparameters
layers = 2
hneurons = [32, act_size]
epochs = 50
batch_size = 5000
lr = 1e-2
hid_act = tf.tanh
out_act = None


# In[6]:


# Build the network
obs_ph = tf.placeholder(shape=(None, obs_size), dtype=tf.float32, name='input')

a1 = tf.layers.dense(obs_ph, units=hneurons[0], activation=hid_act)
logits = tf.layers.dense(a1, units=hneurons[1], activation=None)

# Select the action
actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

# Loss function whose gradient is the policy gradient
weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
action_masks = tf.one_hot(act_ph, act_size)
log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
loss = -tf.reduce_mean(weights_ph * log_probs)

# Make the train op
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


# In[7]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[8]:


def show_state(env):
    plt.figure(3)
    plt.clf()
    a = env.render(mode='rgb_array')
    print (type(a))
    print (a)
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
    
    display.clear_output(wait=True)
    display.display(plt.gcf())


# In[11]:


def train_one_epoch():
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
        
        '''
        if not rendered_once_in_epoch:
            # For notebooks on server (like Colab)
            #show_state(env)
            # For notebooks on local machines
            env.render()
            pass
        '''
        env.render()

        batch_obs.append(obs)
        
        act = sess.run([actions], feed_dict={obs_ph: obs.reshape(1 ,-1)})[0][0]
        
        #print (act)
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
            #print (batch_weights)
            #print (ep_ret, ep_len)
            #print ([ep_ret] * ep_len)
            #input()
            
            # Reset the environment
            obs, done, ep_rews = env.reset(), False, []
            
            rendered_once_in_epoch = True
            
            if batch_size < len(batch_obs):
                break
    
    print(np.array(batch_weights).shape)
    batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              weights_ph: np.array(batch_weights)})
        
        
    return batch_loss, batch_rews, batch_len


# In[12]:


# Training loop
for epoch in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print ('Epoch: {:.3f} Loss: {:.3f} Return: {:.3f} ep_len: {:.3f}'
           .format(epoch+1, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


# In[ ]:


# TensorBoard Setup
'''
get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
get_ipython().system_raw('tensorboard --logdir=./tboard/FrozenLake-v0/ &')
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system('curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')

# In[13]:


env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()

'''
