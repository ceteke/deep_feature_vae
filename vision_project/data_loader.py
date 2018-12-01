import os, tensorflow as tf, math

class DataLoader(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_dataset(self, batch_size):
        filenames = os.listdir(self.base_dir)
        dirs = list(map(lambda x: os.path.join(self.base_dir, x), filenames))
        num_batches = math.ceil(len(dirs)/batch_size)

        dataset = tf.data.Dataset.from_tensor_slices((dirs))
        dataset = dataset.shuffle(len(dirs))
        dataset = dataset.map(self.parse_function, num_parallel_calls=16)
        dataset = dataset.map(self.train_preprocess, num_parallel_calls=16)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset, num_batches

    def parse_function(self, file):
        image_string = tf.read_file(file)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [64, 64]) # FIXME
        return image

    def train_preprocess(self, image):
        return image