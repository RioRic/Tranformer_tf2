import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    """ 
    calculate self-attention
    ----------------
    q: (..., seq_len_q, depth)
    k: (..., seq_len_k, depth)
    v: (..., seq_len_v, depth)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    """ scaled qk """
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_matmul_qk = matmul_qk / tf.math.sqrt(dk)

    """ attention weights """
    attention_weights = tf.nn.softmax(scaled_matmul_qk, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model // self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ split to multi heads """
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        """ (batch_size, seq_len, d_model) """
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        """ (batch_size, num_heads, seq_len, depth)"""
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention:      (batch_size, num_heads, seq_len_q, depth)
        # attention_weights:     (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


class PointWiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, **kwargs):
        super(PointWiseFFN, self).__init__(**kwargs)

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFFN(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None, mask=None):
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.norm1(x + attention_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PointWiseFFN(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training=None, mask=None):
        attention_output, attention_weight_1 = self.mha1(x, x, x)
        attention_output = self.dropout1(attention_output)
        out1 = self.norm1(attention_output + x)

        attention_output, attention_weight_2 = self.mha2(
            enc_output, enc_output, x)
        attention_output = self.dropout2(attention_output)
        out2 = self.norm2(attention_output + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.norm3(ffn_output + out2)

        return out3, attention_weight_1, attention_weight_2
