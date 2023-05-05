import os
import tensorflow as tf
from transform import TransformerModel
from CustomSchedule import CustomSchedule
from config import *
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
from transformers import TFBertModel
model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
import joblib 

if __name__ == "__main__":
    word2Topic = joblib.load('word2Topic.jl')
    list_topic_count = joblib.load('list_topic_count.jl')
    checkpoint_path = "checkpoints"
    learning_rate = CustomSchedule(d_model)
    encoder_vocab_size = tokenizer.vocab_size
    decoder_vocab_size = tokenizer.vocab_size
    pptimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = TransformerModel(
        num_layers, 
        d_model, 
        num_heads, 
        dff,
        encoder_vocab_size, 
        decoder_vocab_size, 
        pe_input=encoder_vocab_size, 
        pe_target=decoder_vocab_size,
        word2Topic=word2Topic,
        list_topic_count=list_topic_count
        )
    transformer.load_weights(checkpoint_path + "/") 
