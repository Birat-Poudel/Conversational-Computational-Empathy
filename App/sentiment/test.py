import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pickle


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


with tf.keras.utils.custom_object_scope(
    {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
    }
):
    model = tf.keras.models.load_model("./SentimentAnalysisTransformer.h5")

text = ["The product quality is worse. I want to return this purchase."]

tokenizer = Tokenizer(num_words=20000, split=" ")

with open("tokenizer2.pickle", "rb") as handle:
    loaded_tokenizer = pickle.load(handle)

text = loaded_tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=150, dtype="int32", value=0)

sentiment = model.predict(text, batch_size=1, verbose=2)[0]

print(sentiment)

if np.argmax(sentiment) == 0:
    print("Negative")
elif np.argmax(sentiment) == 1:
    print("Neutral")
else:
    print("Positive")
