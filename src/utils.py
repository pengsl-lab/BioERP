import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal 
from tflearn.activations import relu

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(tf.cast(x0,tf.float32), tf.cast(W0p,tf.float32)), tf.matmul(tf.cast(x1,tf.float32), tf.cast(W1p,tf.float32)),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(tf.cast(x0,tf.float32), tf.cast(W0p,tf.float32)), tf.matmul(tf.cast(x1,tf.float32), tf.cast(W0p,tf.float32)),transpose_b=True)

def extract_prembedding():
    num = 0
    feature_list = []
    with open('../local_embedding_NeoDTI.txt', 'r') as f:
        bert_feature = f.readline()
        while (bert_feature):
            feature_list.append([])

            layer1_start = bert_feature.find('-1, "' + "values" + '"' + ": [", 31500, 33000) + 15
            layer1_end = bert_feature.find("]},", 39500, 41000)
#             layer2_start = bert_feature.find('-2, "' + "values" + '"' + ": [", 39500, 41000) + 15
#             layer2_end = bert_feature.find("]},", 47500, 49000)

            feature_list[num] = bert_feature[layer1_start:layer1_end].split(", ")
#             feature_list[num].extend(bert_feature[layer2_start:layer2_end].split(", "))


            length = len(feature_list[num])
            for i in range(length):
                feature_list[num][i] = float(feature_list[num][i])
            num += 1
            bert_feature = f.readline()

    return feature_list
    
    
def extract_finembedding():
    num = 0
    feature_list = []
    with open('../global_embedding_NeoDTIpath.txt', 'r') as f:
        bert_feature = f.readline()
        while (bert_feature):
            feature_list.append([])
            layer1_start = bert_feature.find('-1, "' + "values" + '"' + ": [", 31500, 33000) + 15
            layer1_end = bert_feature.find("]},", 39500, 41000)
            # layer2_start = bert_feature.find('-2, "' + "values" + '"' + ": [", 39500, 41000) + 15
            # layer2_end = bert_feature.find("]},",  47500, 49000)


            feature_list[num] = bert_feature[layer1_start:layer1_end].split(", ")
            # feature_list[num].extend(bert_feature[layer2_start:layer2_end].split(", "))

            length = len(feature_list[num])
            for i in range(length):
                feature_list[num][i] = float(feature_list[num][i])
            num += 1
            bert_feature = f.readline()
    return feature_list


def extract_finalembedding():
    prembedding = extract_prembedding()
    finembedding = extract_finembedding()

    finalembedding = []
    length = len(finembedding)
    for i in range(length):
        finalembedding.append([])
        finalembedding[i].extend(prembedding[i])
        finalembedding[i].extend(finembedding[i])

    drug_feature = np.array(finalembedding)
    return drug_feature

def add_finalembedding():
    prembedding = extract_finembedding()
    finembedding = extract_prembedding()
    finalembedding = np.array(prembedding)*0.5 + np.array(finembedding)*0.5
    return finalembedding
