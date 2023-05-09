import tensorflow as tf
class BertLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer, integrating BERT from tf-hub
    """
    def __init__(self, bert):
        super(BertLayer, self).__init__()
        self.bert = bert       

    def call(self, inputs):
        # doc_input_ids = []
        # doc_attention_mask = []
        # doc_token_type_ids = []
        # tok = self.tokenizer(inputs,return_tensors='tf')
        # tok = dict(inputs)
        # doc_input_ids = [tf.keras.cast(x, dtype="int32") for x in tok["input_ids"]]
        # doc_attention_mask = [tf.keras.cast(x, dtype="int32") for x in tok["attention_mask"]]
        # doc_token_type_ids = [tf.keras.cast(x, dtype="int32") for x in tok["token_type_ids"]]
                
        # bert_inputs = dict(input_ids=doc_input_ids, input_mask=doc_attention_mask, segment_ids=doc_token_type_ids)
        result = self.bert(inputs)["last_hidden_state"]
        return result