import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    'Arduino_Image_Training_Set/train',
    'Arduino_Image_Training_Set/train',
    ['Battery1', 'Battery2', 'Battery3', 'Battery4']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'Arduino_Image_Training_Set/validate',
    'Arduino_Image_Training_Set/validate',
    ['Battery1', 'Battery2', 'Battery3', 'Battery4']
)

print(train_data)
print(val_data)

spec = model_spec.get('efficientdet_lite0')

model = object_detector.create(train_data, model_spec=spec, batch_size=32, train_whole_model=True, epochs=25, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='arduino.tflite')
