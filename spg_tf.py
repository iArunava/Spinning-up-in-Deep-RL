import numpy as np
import tensorflow as tf

def train_with_tf(FLAGS, act_size):
    # Network Hyperparameters
    layers = 2
    hneurons = [32, act_size]
    epochs = FLAGS.epochs
    batch_size = 5000
    lr = 1e-2
    hid_act = tf.tanh
    out_act = None

    graph_path = FLAGS.graph_path
    if graph_path[0] != '/':
        graph_path += '/'

    # Build the network
    obs_ph = tf.placeholder(shape=(None, obs_size), dtype=tf.float32, name='input')

    a1 = tf.layers.dense(obs_ph, units=hneurons[0], activation=hid_act)
    logits = tf.layers.dense(a1, units=hneurons[1], activation=None)

    # Select the action
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1, name='output')

    # Loss function whose gradient is the policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, act_size)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

    # Make the train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training Loop Parameters
    ckpt_interval = 5
    save_path = './ckpt_path/'
    restore = False
    ckpt_num = 45

    # Training loop
    if restore and ckpt_num != None:
        saver.restore(sess, save_path + 'spg_ckpt{}.ckpt'.format(ckpt_num))
        print ('[INFO]Model Restored!!')

    for epoch in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(sess)
        print ('[INFO]Epoch: {:.3f} Loss: {:.3f} Return: {:.3f} ep_len: {:.3f}'
               .format(epoch+1, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

        if (epoch+1) % ckpt_interval == 0:
            print ('[INFO]Saving Checkpoint...')
            curr_save_path = saver.save(sess, save_path + 'spg_ckpt{}.ckpt'.format(epoch+1))
            print ('[INFO]Session saved Successfully!!')
            print ('[INFO]Checkpoint saved at: {}'.format(curr_save_path))
            print ('*************************************************')

    # save the weights and graph
    if True:
        print ('[INFO]Saving the graph and weights...')
        save_graph(sess, tf.get_default_graph(), graph_path + env_name + '-graph.pb')
        print ('[INFO]Saved Successfully!!')

    sess.close()

def test_with_tf(FLAGS):
    graph = load_graph(FLAGS.graph)

    # Test the network
    input_layer = 'import/' + FLAGS.input_layer
    output_layer = 'import/' + FLAGS.output_layer

    input_op = graph.get_operation_by_name(input_layer)
    output_op = graph.get_operation_by_name(output_layer)

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        obs = env.reset()
        done = False
        ep_rews = 0

        while not done:
            act = sess.run([output_op.outputs[0]], feed_dict={input_op.outputs[0]: obs.reshape(1, -1)})

            obs, rewards, done, info = env.step(act[0][0])

            ep_rews += rewards

        print ('Test Episode Rewards: {}'.format(ep_rews))
