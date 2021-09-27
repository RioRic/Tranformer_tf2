import tensorflow as tf
from ..model.basemodel import Encoder, Decoder


class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(num_layers, d_model,
                               num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, d_model,
                               num_heads, dff, target_vocab_size, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar):
        enc_output = self.encoder(inp)

        dec_output, attention_weights = self.decoder(tar, enc_output)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
