import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices("GPU")

print(gpu)
