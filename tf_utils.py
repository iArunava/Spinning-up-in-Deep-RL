import numpy as np
import tensorflow as tf
import gym

# Saving the weights along with the graph
def save_graph(sess, graph, graph_name):
    output_graph_def = graph_util.convert_variables_to_constants(
                        sess, graph.as_graph_def(), ['output'])
    with gfile.FastGFile(graph_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return

# Load the graph
def load_graph(model_file):
    print ('[INFO]Loading Model...')
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    print ('[INFO]Reading model file...')
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def)

    print ('[INFO]Model Loaded Successfully!!')
    return graph

def save_ckpt(ckpt_name):
