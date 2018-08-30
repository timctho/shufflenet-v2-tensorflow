import tensorflow as tf
from net import ShuffleNetV2

if __name__ == '__main__':
    # Create input placeholder
    input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])

    # Create model
    model = ShuffleNetV2(input, 1001, model_scale=1.0, is_training=True)

    # Get model inference result
    print(model.output) # shape = [None, 1001]


