import os
import numpy as np
from PIL import Image

from vaik_count_pb_inference.pb_model import PbModel

input_saved_model_dir_path = os.path.expanduser('~/.vaik-count-pb-trainer/output_model/2023-11-25-16-53-22/step-1000_batch-16_epoch-9_loss_0.0493_val_loss_0.2837')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-count-dataset/valid/valid_000000011_raw.png')).convert('RGB'))

classes = ('zero', 'one', 'two')
model = PbModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([image], batch_size=1)