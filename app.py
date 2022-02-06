# this runs well on TF 2.6.1 docker container
import tensorflow as tf
from ai_benchmark import AIBenchmark

if gpu := tf.config.experimental.list_physical_devices("GPU"):
    bm = AIBenchmark()
else:
    print("No GPU found")
    bm = AIBenchmark(use_CPU=True)

results = bm.run()
