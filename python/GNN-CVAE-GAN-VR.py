#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate, Embedding,Activation, LeakyReLU, ELU
import json

current = os.getcwd()

# In[18]:


sequence_length = 8
batch_size = 64
vocab_size = 1000
embed_dim = 256
latent_dim = 2048
num_heads = 8
type_size = 8
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential([layers.Dense(dense_dim, activation="LeakyReLU"), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True
    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential([layers.Dense(latent_dim, activation="LeakyReLU"), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:,tf.newaxis, :], dtype="int32")#
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],axis=0,)
        return tf.tile(mask, mult)
    
type_input = Input(shape=(8), dtype="int64", name="type_inputs")
door_input = Input(shape=(2), dtype="float32", name="door_inputs")
bound_input = Input(shape=(64), dtype="float32", name="bound_inputs")
door_embedding = layers.Dense((256), activation="LeakyReLU")(door_input)
bound_embedding = layers.Dense((512), activation="LeakyReLU")(bound_input)
door_embedding = layers.RepeatVector(8)(door_embedding)
bound_embedding = layers.RepeatVector(8)(bound_embedding)
x = PositionalEmbedding(sequence_length, type_size, embed_dim)(type_input)
x = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
x = Concatenate()([x,door_embedding, bound_embedding])
encoder_outputs = layers.Dense((256), activation="LeakyReLU")(x)
encoder = Model([type_input,door_input,bound_input], encoder_outputs)
decoder_inputs = Input(shape=(8,2), dtype="int64", name="decoder_inputs")
decoder_embedding = layers.Dense((512), activation="LeakyReLU")(decoder_inputs)
decoder_pooling = layers.GlobalAveragePooling1D(data_format="channels_first")(decoder_embedding)
encoded_seq_inputs = Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_pooling)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(2*vocab_size, activation="LeakyReLU")(x)
decoder_outputs = layers.Reshape((8,2, vocab_size))(decoder_outputs)
decoder_outputs = Activation('softmax')(decoder_outputs)
decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs,name="decoder_outputs")
decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = Model(
    [decoder_inputs,type_input,door_input,bound_input], [decoder_outputs], name="transformer")


# In[32]:


transformer.load_weights(os.path.join(current,'pretrained_models/location_predictor_trained/transformer1000_10ep'))
type_predictor_trained = load_model(os.path.join(current,'pretrained_models/type_predictor_trained/type_predictor_trained'), compile=False)
location_predictor_trained = transformer
size_predictor_trained = load_model(os.path.join(current,'pretrained_models/type_predictor_trained/size_predictor_trained'), compile=False)
edge_predictor_trained = load_model(os.path.join(current,'pretrained_models/type_predictor_trained/edge_predictor_trained'), compile=False)
ratio_predictor_trained = load_model(os.path.join(current,'pretrained_models/type_predictor_trained/ratio_predictor_trained'), compile=False)


# In[50]:


bound1 = np.array([[[115, 115, 115, 115, 115, 115, 115, 115],
        [115,  30, 172, 172, 172, 172,  77, 115],
        [115, 197,  86,   6,   6, 199,  77, 115],
        [115, 197,  86, 115, 115, 199,  77, 115],
        [115, 197,  86, 115,   6, 199, 215, 115],
        [115, 197,  86, 115, 108, 192, 215, 115],
        [115, 197,  86, 128, 108, 178, 115, 115],
        [115, 115, 235, 235, 235, 115, 115, 115]]]).reshape(1,64)*(1/255.)
bound2 = np.array([[0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.45490196,
        0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.10980392,
        0.0627451 , 0.90588235, 0.90588235, 0.0627451 , 0.33333333,
        0.45490196, 0.45490196, 0.91372549, 0.33333333, 0.33333333,
        0.33333333, 0.89019608, 0.94901961, 0.45490196, 0.45490196,
        0.91372549, 0.2745098 , 0.45490196, 0.45490196, 0.89019608,
        0.2745098 , 0.45490196, 0.45490196, 0.91372549, 0.2745098 ,
        0.45490196, 0.45490196, 0.89019608, 0.2745098 , 0.45490196,
        0.45490196, 0.91372549, 0.33333333, 0.33333333, 0.33333333,
        0.89019608, 0.94901961, 0.45490196, 0.45490196, 0.38431373,
        0.08235294, 0.79215686, 0.79215686, 0.79215686, 0.2745098 ,
        0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.45490196,
        0.45490196, 0.45490196, 0.45490196, 0.45490196]])
bound3 = np.array([[0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.45490196,
        0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.68627451,
        0.99607843, 0.33333333, 0.94901961, 0.74509804, 0.50588235,
        0.45490196, 0.45490196, 0.68627451, 0.20784314, 0.1254902 ,
        0.1254902 , 0.85490196, 0.50588235, 0.45490196, 0.45490196,
        0.68627451, 0.08235294, 0.2745098 , 0.45490196, 0.94901961,
        0.50588235, 0.45490196, 0.45490196, 0.68627451, 0.08235294,
        0.2745098 , 0.45490196, 0.94901961, 0.50588235, 0.45490196,
        0.45490196, 0.68627451, 0.20784314, 0.50588235, 0.50588235,
        0.85490196, 0.50588235, 0.45490196, 0.45490196, 0.24705882,
        0.86666667, 0.33333333, 0.33333333, 0.12941176, 0.45490196,
        0.45490196, 0.45490196, 0.45490196, 0.45490196, 0.45490196,
        0.45490196, 0.45490196, 0.45490196, 0.45490196]])
door1 = np.expand_dims(np.array([0.75,0.4]),axis=0)
door2 = np.array([[0.46862745, 0.21372549]])
door3 = np.array([[0.34313725, 0.78235294]])
boundlist = [bound1,bound2,bound3]
doorlist = [door1,door2,door3]


# In[26]:


max_decoded_sentence_length = 8
def decode_sequence(typein,doorin,boundin):
    decoded_sentence = np.expand_dims(np.array([[999,999],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]),axis=0)
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = decoded_sentence[:,:-1]
        predictions = location_predictor_trained([tokenized_target_sentence,typein,doorin,boundin])
        sampled_token_index = np.array([np.argmax(predictions[0, i,0, :]),np.argmax(predictions[0, i,1, :])])
        decoded_sentence[:,i+1] = sampled_token_index
        if sampled_token_index[0] == 0:
            break
    return decoded_sentence


# In[29]:


# latent = np.linspace(0*np.ones((32)), 1*np.ones((32)), num=5)
with open(os.path.join(current,'graph.json'), 'r') as f:
    datagraph = json.load(f)
with open(os.path.join(current,'boundary.json'), 'r') as g:
    data_boundary = json.load(g)
bound = boundlist[data_boundary]
door = doorlist[data_boundary]
latent = 0.5*np.ones((32))
types = []
locations = []
sizes = []
edges = []
ratios = []
latents = np.expand_dims(i, axis=0)
get_type = type_predictor_trained.predict([latents,bound])
types_ind = np.argmax(get_type, axis=2)
types_ind[types_ind==6] = 8
types_ind[types_ind==1] = 2
types_ind[types_ind==0] = 8
types_ind = np.sort(types_ind)
types_ind[types_ind==8] = 0
node_type_seq = [5,0,4,4,4]
types.append(types_ind)
pad_type = np.array(np.insert(types_ind,6,0,axis = 1),dtype=np.int32)
get_location = np.expand_dims(decode_sequence(pad_type,door,bound)[0,1:-1,:]/1000, axis=0)#np.expand_dims(loca, axis=0)#NodeList[ind][:,:2]
locations.append(get_location)
get_size = size_predictor_trained.predict([get_location,get_type,bound])
sizes.append(get_size)
get_edge = edge_predictor_trained.predict([get_size,get_location,get_type,bound])
edges.append(get_edge)
get_ratio = ratio_predictor_trained.predict([get_edge,get_size,get_location,get_type,bound])
ratios.append(get_ratio)
edges_ind = np.argmax(edges, axis=4)


    





                
        

