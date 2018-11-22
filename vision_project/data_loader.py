import os, tensorflow as tf

class DataLoader(object):
    def __init__(self, base_dir, n_dataset=20):
        self.base_dir = base_dir
        self.image_shape = [218, 178, 3]
        self.tf_record_prefix = 'data'
        self.n_dataset = n_dataset

    def parser(self, record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, self.image_shape)
        image = tf.cast(image, tf.float32)

        return image

    def load_dataset(self, batch_size):
        filenames = os.listdir(self.base_dir)
        dirs = list(map(lambda x: os.path.join(self.base_dir, x), filenames))
        dataset = tf.data.Dataset.from_tensor_slices((dirs))
        dataset = dataset.shuffle(len(dirs))
        dataset = dataset.map(self.parse_function, num_parallel_calls=16)
        dataset = dataset.map(self.train_preprocess, num_parallel_calls=16)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def parse_function(self, file):
        image_string = tf.read_file(file)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [64, 64]) # FIXME
        return image

    def train_preprocess(self, image):
        return image