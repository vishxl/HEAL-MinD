import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D,  GRU, Dense, Concatenate, Attention, Dropout, Flatten
)
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.backend as K

from transformers import BertTokenizer, TFBertModel

from sklearn.metrics import classification_report

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

MAX_SEQ_LEN = 128  
EMBEDDING_DIM = 300  
VOCAB_SIZE = 600000 
FILTERS = 256
KERNEL_SIZE = 5 
DENSE_UNITS = 32
AUX_FEATURE_LEN = 26 

EPOCH = 5
BATCH_SIZE = 64

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "label" 

bert_model_path = "/Users/vishalksingh/models/bert-base-uncased/"

datasets = ['synthetic','CFPB','huggingface']
data_name = datasets[1]

path = f"dataset/final_dataset/split_data/{data_name}"

df_train = pd.read_csv(path+"_train.csv")
df_test = pd.read_csv(path+"_test.csv")
df_val = pd.read_csv(path+"_val.csv")

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)

df_train = df_train.dropna()
df_test = df_test.dropna()
df_val = df_val.dropna()

for column in df_val.columns:
    if df_val[column].isnull().any():
        print(f"Column '{column}' contains NaN values.")
    else:
        print(f"Column '{column}' does not contain NaN values.")

# Auxiliary Features
aux_path = f"dataset/Feature_Vectors (Normalised)_Results/{data_name}"
aux_df_train = pd.read_csv(aux_path+"_train_fvect27_all_final.csv")
aux_df_test = pd.read_csv(aux_path+"_test_fvect27_all_final.csv")
aux_df_val = pd.read_csv(aux_path+"_val_fvect27_all_final.csv")

# F1-Score 
def cust_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Proposed Model: HEAL-MinD

bert_model = TFBertModel.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

def bert_tokenize_texts(texts, tokenizer, max_len):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf"
    )
    return encodings['input_ids'], encodings['attention_mask']

bert_input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="BERT_Input_Ids")
bert_attention_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="BERT_Attention_Mask")

auxiliary_input = Input(shape=(AUX_FEATURE_LEN,), name="auxiliary_input")

def load_glove_embeddings(glove_path, vocab_size, embedding_dim):
    embeddings_index = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    return embedding_matrix

embedding_matrix = load_glove_embeddings("./dataset/glove.6B.300d.txt", VOCAB_SIZE, EMBEDDING_DIM)

text_input = Input(shape=(MAX_SEQ_LEN,), dtype=tf.string, name="text_input")

#BERT 
def bert_encode(texts, tokenizer, max_len):
    encodings = tokenizer(texts.tolist(), max_length=max_len, truncation=True, padding=True, return_tensors="tf")
    return encodings["input_ids"], encodings["attention_mask"]

bert_input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="bert_input_ids")
bert_attention_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="bert_attention_mask")
bert_output = bert_model(bert_input_ids, attention_mask=bert_attention_mask)[0]

# GloVe
glove_input = Input(shape=(MAX_SEQ_LEN,), name="glove_input")
glove_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(glove_input)

# BERT and GloVe 
combined_embedding = Concatenate()([bert_output, glove_embedding])

cnn_output = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu")(combined_embedding)

# BiGRU with Attention
forward_gru = GRU(128, return_sequences=True, go_backwards=False)(cnn_output)
forward_attention = tf.keras.layers.Attention()([forward_gru, forward_gru])

backward_gru = GRU(128, return_sequences=True, go_backwards=True)(cnn_output)
backward_attention = tf.keras.layers.Attention()([backward_gru, backward_gru])

combined_attention= tf.keras.layers.Concatenate()([forward_attention, backward_attention])
combined_attention = tf.keras.layers.Attention()([combined_attention, combined_attention])

auxiliary_input = Input(shape=(26,), name="auxiliary_input")

hybrid_attention = Concatenate()([Flatten()(combined_attention), auxiliary_input])

dense_output = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l2(0.01))(hybrid_attention)
dense_output = Dropout(0.3)(dense_output)
final_output = Dense(1, activation='sigmoid')(dense_output)

model = Model(
    inputs=[bert_input_ids, bert_attention_mask, glove_input, auxiliary_input],
    outputs=final_output,
    name="heal_mind_model"
)

for layer in model.layers:
    if 'tf_bert_model' in layer.name:
        print(layer.name)
        layer.trainable = False

model.compile(loss="binary_crossentropy", optimizer='Adadelta', metrics=['accuracy', cust_f1])

model.summary()


glove_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
glove_tokenizer.fit_on_texts(df_train[TEXT_COLUMN]) 

def prepare_bert_inputs(texts, tokenizer, max_len):
    input_ids, attention_masks = bert_tokenize_texts(texts, tokenizer, max_len)
    return input_ids, attention_masks

def prepare_inputs(data, aux_df):
    # GloVe
    glove_sequences = glove_tokenizer.texts_to_sequences(data[TEXT_COLUMN])
    glove_padded = pad_sequences(glove_sequences, maxlen=MAX_SEQ_LEN, padding="post")
    
    # BERT inputs
    bert_input_ids, bert_attention_masks = prepare_bert_inputs(data[TEXT_COLUMN], bert_tokenizer, MAX_SEQ_LEN)
    
    aux_features = aux_df.values  
    labels = data[LABEL_COLUMN].values
    
    return bert_input_ids, bert_attention_masks, glove_padded, aux_features, labels

train_inputs = prepare_inputs(df_train, aux_df_train)
val_inputs = prepare_inputs(df_val, aux_df_val)
test_inputs = prepare_inputs(df_test, aux_df_test)

train_bert_ids, train_bert_masks, train_glove, train_aux, train_labels = train_inputs
val_bert_ids, val_bert_masks, val_glove, val_aux, val_labels = val_inputs
test_bert_ids, test_bert_masks, test_glove, test_aux, test_labels = test_inputs

print("Training class distribution:", Counter(train_labels))
print("Validation class distribution:", Counter(val_labels))

# Train
history = model.fit(
    [train_bert_ids, train_bert_masks, train_glove, train_aux], 
    train_labels,
    validation_data=([val_bert_ids, val_bert_masks, val_glove, val_aux], val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    # callbacks=[tf.keras.callbacks.EarlyStopping(
    #            patience=15,
    #              min_delta=0.01,
    #              baseline=0.9,
    #              mode='auto',
    #              monitor='val_output_accuracy',
    #              restore_best_weights=True,
    #              verbose=1)
)   

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(f"dataset/{data_name}_history_e{EPOCH}.csv")

#Evaluate
test_loss, test_accuracy, test_f1= model.evaluate(
    [test_bert_ids, test_bert_masks, test_glove, test_aux], 
    test_labels
)
print(f"\n\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")

test_pred = model.predict([test_bert_ids, test_bert_masks, test_glove, test_aux])
print(test_pred.shape)

pred_labels = [int(np.round(pred.mean())) for pred in test_pred]  

print(f"Epoch: {EPOCH} \nBatch Size: {BATCH_SIZE}")

df_test['predicted_labels'] = pred_labels
df_test.to_csv(f"dataset/{data_name}_predicted.csv")

print(classification_report(test_labels, pred_labels))

pd.DataFrame(classification_report(test_labels, pred_labels, output_dict=True)).transpose().to_csv(f"dataset/{data_name}_report_e{EPOCH}.csv")