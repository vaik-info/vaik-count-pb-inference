# vaik-count-pb-inference

```python
import os
import numpy as np
from PIL import Image

from vaik_count_pb_inference.pb_model import PbModel

input_saved_model_dir_path = os.path.expanduser('~/.vaik-count-pb-trainer/output_model/2023-11-25-17-49-54/step-1000_batch-16_epoch-20_loss_0.0193_val_loss_0.0175')
image1 = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-count-dataset/valid/valid_000000002_raw.png')).convert('RGB'))

classes = ('zero', 'one', 'two')
model = PbModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([image1], batch_size=1)
```