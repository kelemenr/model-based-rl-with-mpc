import tensorflow as tf


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, name, value, step):
        with self.writer.as_default():
            tf.summary.scalar(name=name, data=value, step=step)