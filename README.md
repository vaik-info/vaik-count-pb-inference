# vaik-count-pb-inference

Inference by count PB model

## Install

```shell
pip install git+https://github.com/vaik-info/vaik-count-pb-inference.git
```

## Usage
### Example

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

## Output

- output
```
[{'count': [4.8042, 2.8184, 0.9538], 'cam': array([[[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        ・・・], dtype=float32)},
        ・・・]
```

- raw_pred

```
(array([[0.9881473 , 0.        , 1.9089862 ],
       ・・・], dtype=float32), array([[[[0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.1        ],
        ...,], dtype=float32))
```
