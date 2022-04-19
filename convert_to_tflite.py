import torch
import onnx
import torch
import tensorflow as tf
from onnx_tf.backend import prepare

# torch - 1.6.0
# torchvision - 0.7.0
# onnx - 1.7.0
# tensorflow - 2.2.0
# tensorflow-addons - 0.11.2
# onnx-tf - 1.8.0


# File names ( change as needed )
onnx_model_name ='spectrogram.onnx'
tf_model_name = 'spectrogram_pb'
tf_lite_model_name = 'spectrogram.tflite'


# Load the ONNX file
model = onnx.load(onnx_model_name)

# ONNX model to Tensorflow
tf_rep = prepare(model)

#Tensorflow Model 
tf_rep.export_graph(tf_model_name)

# from tensorflow.python.platform import gfile
# GRAPH_PB_PATH = tf_model_name + "/saved_model.pb"
# with tf.compat.v1.Session() as sess:
#    print("load graph")
#    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#        graph_def = tf.compat.v1.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf.compat.v1.import_graph_def(graph_def, name='')
#    graph_nodes=[n for n in graph_def.node]
#    names = []
#    for t in graph_nodes:
#       names.append(t.name)
#    print(names)

# TF to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_name)
# converter = tf.v1.lite.TFLiteConverter.from_frozen_graph(
# 	input_shapes = {"input" : [1, 1, 128, 64]},
#     graph_def_file = tf_model_name + "/saved_model.pb", 
#     input_arrays = [tf_rep.inputs[0]], #Appropritaely choose Model Input(s) - ( change as needed if multiple inputs )
#     output_arrays = ["mask"] # Appropritaely choose Model Output(s) - ( change as needed if multiple outputs )
# )

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

#Saving tflite model
with tf.io.gfile.GFile(tf_lite_model_name, 'wb') as f:
    f.write(tflite_model)