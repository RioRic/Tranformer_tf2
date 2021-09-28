import tensorflow as tf
from ..model.basemodel import Encoder, Decoder
from ..model.baselayer import EncoderLayer
from einops.layers.tensorflow import Rearrange


class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, hidden_dim, dropout, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.layers = [EncoderLayer(
            d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)]

    def call(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class ViT(tf.keras.layers.Layer):
    def __init__(self, num_class, channels, image_size, patch_size, num_layers, d_model, num_heads, hidden_dim, dropout, emb_dropout, **kwargs):
        super(ViT, self).__init__(**kwargs)
        H, W = image_size
        PH, PW = patch_size

        assert H % PH == 0 and W % PW == 0

        num_patch = H * W // (PH * PW)

        patch_dim = channels * PH * PW

        """ einops to patch """
        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PH, p2=PW)

        self.patch_embedding = tf.keras.layers.Dense(patch_dim, d_model)
        """ position embedding """
        self.pos_embedding = tf.Variable(
            tf.random.normal(shape=[1, num_patch + 1, d_model]))
        """ class token """
        self.cls_token = tf.Variable(tf.random.normal(shape=[1, 1, d_model]))

        self.dropout = tf.keras.layers.Dropout(emb_dropout)

        self.transformer = Transformer(
            num_layers, d_model, num_heads, hidden_dim, dropout)

        self.LayerNorm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(num_class)

    def call(self, img):
        x = self.rearrange(img)
        x = self.patch_embedding(x)

        x = tf.concat([self.cls_token, x], axis=1)
        x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)

        x = tf.reduce_mean(x, axis=1)

        x = self.LayerNorm(x)
        x = self.dense(x)

        return x
