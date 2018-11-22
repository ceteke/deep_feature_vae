import numpy as np, os, tensorflow as tf
from tqdm import tqdm
from keras_preprocessing import image

class DataLoader(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.image_shape = [218, 178, 3]
        self.tf_record_path = 'data.tfrecord'

    def parser(self, record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, self.image_shape)
        image = tf.cast(image, tf.float32)

        return image

    def load_dataset(self):
        self.dataset = tf.data.TFRecordDataset(self.tf_record_path)
        self.dataset = self.dataset.map(self.parser, num_parallel_calls=16)

    def load_image(self, dir):
        dir = os.path.join(self.base_dir, dir)
        img = image.img_to_array(image.load_img(dir))
        return img

    def save_tf_record(self, writer, image):
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())

    def prepare_images(self):
        dirs = os.listdir(self.base_dir)
        with tf.python_io.TFRecordWriter(self.tf_record_path) as writer:
            for dir in tqdm(dirs):
                img = self.load_image(dir)
                self.save_tf_record(writer, img)