import pandas as pd
import tensorflow as tf
import json

tf.logging.set_verbosity(tf.logging.INFO)

# input_column = tf.feature_column.numeric_column(key="input", shape=(64,))
NUMBERS = range(64)

feature_cols = [tf.feature_column.numeric_column(str(k)) for k in NUMBERS]

input = []
output = []

with open('output') as data:
    for line in data:
        result = json.loads(line)
        expected = int(1000 * result['output'][0])
        if expected != 999 and expected != -999:
            input.append(result['input'])
            output.append(expected)

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                    hidden_units=[512, 256, 256, 256, 256, 256, 256, 256],
                                    model_dir="/tmp/chess_dude/512-256-256-256-256-256-256-256")

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  get_n_item = lambda n: map(lambda array: array[n], data_set)
#   print(get_n_item(1))
#   return 1
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({str(k): get_n_item(k) for k in NUMBERS}),
      y = pd.Series(output),
      num_epochs=num_epochs,
      shuffle=shuffle)

regressor.train(input_fn=get_input_fn(input), steps=5000)
