import tensorflow_datasets as tfds
import tensorflow as tf
import os
from data import make_dataset

class TrainConfig(object):
  embedding_dim = 256
  rnn_units = 64
  buffer_size = 10000
  batch_size = 40
  take_size = 1200
  epochs = 10
  validation_steps = 30
  vocab_size = 5

def train(dataset, checkpoint_dir, config):
  train_dataset = dataset.skip(config.take_size)
  train_dataset = train_dataset.shuffle(config.buffer_size).padded_batch(config.batch_size, padded_shapes=([None],[]))
  
  test_dataset = dataset.take(config.take_size)
  test_dataset = test_dataset.padded_batch(config.batch_size, padded_shapes=([None],[]))
  
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(config.vocab_size, config.embedding_dim),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units)),
      tf.keras.layers.Dense(config.rnn_units, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
  
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
  
  history = model.fit(train_dataset, epochs=config.epochs,
                      validation_data=test_dataset, 
                      validation_steps=config.validation_steps,
  					  callbacks=[checkpoint_callback])

if __name__ == "__main__":
  config = TrainConfig
  dataset, encoder = make_dataset('train.txt', ['name','gender'], 'gender')
  config.vocab_size = encoder.vocab_size
  train(dataset, './training_checkpoints', config)