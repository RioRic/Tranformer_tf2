import tensorflow as tf

from ..model.basemodel import EncoderLayer, DecoderLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_size, dropout_rate=0.1, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)

        self.num_layers = num_layers

        self.d_model = d_model
        self.num_heads = num_heads

        self.embedding = tf.keras.layers.Embedding(input_size, d_model)

        self.enc_layers = [EncoderLayer(
            d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, traget_vocab_size, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.num_layers = num_layers

        self.d_model = d_model
        self.num_heads = num_heads

        self.embedding = tf.keras.layers.Embedding(traget_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(
            d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block_1, block_2 = self.dec_layers[i](x)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block_1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block_2

        return x, attention_weights
