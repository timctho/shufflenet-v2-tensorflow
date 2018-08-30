# Shufflenet-v2-tensorflow
Tensorflow implementation of ECCV 2018 paper ShuffleNet V2. [[Paper]](https://arxiv.org/abs/1807.11164)

<p align="center">
    <img src="https://github.com/timctho/shufflenet-v2-tensorflow/raw/master/shuffle-block.png">
</p>

# Support architectures
  * ShuffleNetV2 with 0.5, 1.0, 1.5, 2.0 channel multipliers

# Usage
```python
import tensorflow as tf
from net import ShuffleNetV2


if __name__ == '__main__':
    # Create input placeholder
    input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])

    # Create model
    model = ShuffleNetV2(input, 1001, model_scale=1.0)

    # Get model inference result
    print(model.output) # shape = [None, 1001]

```
