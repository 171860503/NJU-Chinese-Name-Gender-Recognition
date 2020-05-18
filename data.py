import tensorflow_datasets as tfds
import tensorflow as tf


SELECT_COLUMNS = ['name','gender']
LABEL_COLUMN = 'gender'

def make_dataset(filename, select_columns, label_name):
  dataset = tf.data.experimental.make_csv_dataset(filename, 1, select_columns=select_columns, label_name=label_name, num_epochs=1)
  dataset = dataset.map(lambda ex,i: (ex['name'][0],i[0]))
  
  vocabulary_set = set()
  for text_tensor, _ in dataset:
    some_tokens = list(text_tensor.numpy().decode())
    vocabulary_set.update(some_tokens)
  
  vocab_size = len(vocabulary_set)
  encoder = tfds.features.text.SubwordTextEncoder(vocabulary_set)
  
  def encode(text_tensor):
    encoded_text = encoder.encode(text_tensor.numpy())
    return [encoded_text]
  
  def encode_map_fn(text, label):
    encoded_text = tf.py_function(encode, inp=[text], Tout=tf.int64)
    return encoded_text, label
  
  dataset = dataset.map(encode_map_fn)
  return dataset, encoder