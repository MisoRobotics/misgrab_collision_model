import tensorflow as tf
import pandas as pd
import numpy as np
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set option to show full column width
pd.set_option('display.max_colwidth', None)


class Predictor:
    def __init__(self, model_path: str, expected_columns: list[str], high_thresh: float, low_thresh: float) -> None:
        self.model_path = model_path
        self.serving_default = self.load_tf_model(self.model_path)
        self.expected_columns = expected_columns
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def load_tf_model(self, model_path: str):
        model = tf.saved_model.load(model_path)
        # Accessing the serving_default function to make a prediction
        serving_default = model.signatures['serving_default']
        return serving_default
    
    
    def make_pred(self, input_data: pd.DataFrame) -> bool:
        start_time = time.time()
        if isinstance(input_data, pd.DataFrame):
            # Reorder DataFrame columns to match the expected order
            input_data = input_data[self.expected_columns]
            # Convert DataFrame to numpy array
            input_data = input_data.to_numpy().astype(np.float32)
        input_dict = {'dense_input': tf.constant(input_data)}
        prediction = self.serving_default(**input_dict)
        probability = prediction['dense_4'].numpy().item()
        return probability>=self.high_thresh, probability>=self.low_thresh, probability
        