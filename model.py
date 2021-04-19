import sys
import tensorflow as tf
from tensorflow import keras

import os
modulepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"transformer")
sys.path.append(modulepath)

from transformer import get_encoders
from layer_normalization import LayerNormalization
from tensorflow.keras import backend as K

def get_model(transformer_num,head_num,feed_forward_dim,dropout_rate,max,istransfer,modelname):
    input_txt = keras.layers.Input(shape=(max,301))
    embeded = keras.layers.Dropout(dropout_rate)(input_txt)
    embeded = LayerNormalization(name='Embedding-Norm')(embeded)

    transformed = get_encoders(
            encoder_num=transformer_num,
            input_layer=embeded,
            head_num=head_num,
            hidden_dim=feed_forward_dim,
            dropout_rate=dropout_rate,
        )
    Adder = keras.layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    output = Adder(transformed)
    output = keras.layers.Dense(256,activation='relu')(output)
    output = keras.layers.Dense(64)(output)
    output = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(output)
    model = keras.models.Model(input_txt,output)
    if istransfer==True:
        model.load_weights(modelname)
        print("load model finished")
        new = keras.layers.Dense(256,activation='relu')(model.layers[-4].output)
        new = keras.layers.Dense(64)(new)
        new = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(new)
        model = keras.models.Model(input_txt,new)
    return model
