import numpy as np, os, tensorflow as tf, threading
from tqdm import tqdm
from keras_preprocessing import image

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

    def prepare_images(self):
        dirs = os.listdir(self.base_dir)
        n_imgs = len(dirs)
        img_per_thread = int(n_imgs/self.n_dataset)
        threads = []

        for i in range(0, n_imgs, img_per_thread):
            upper = min(n_imgs, i+img_per_thread)
            thread_dirs = list(map(lambda x: os.path.join(self.base_dir, x), dirs[i:upper]))
            data_id = int(i / img_per_thread)
            data_file = '{}_{}.tfrecord'.format(self.tf_record_prefix, data_id)
            thread = DataWriter(data_id, thread_dirs, data_file)
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

    def load_dataset(self):
        for i in range(self.n_dataset):
            tf_record_path = '{}_{}.tfrecord'.format(self.tf_record_prefix, i)
            dataset = tf.data.TFRecordDataset(tf_record_path)
            dataset = dataset.map(self.parser, num_parallel_calls=16)
            if i == 0:
                self.dataset = dataset
            else:
                self.dataset = self.dataset.concatenate(dataset)


class DataWriter(threading.Thread):
    def __init__(self, id, dirs, tf_record_path):
        threading.Thread.__init__(self)
        self.id = id
        self.dirs = dirs
        self.tf_record_path = tf_record_path

    def load_image(self, dir):
        img = image.img_to_array(image.load_img(dir))
        return img

    def save_tf_record(self, writer, image):
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())

    def run(self):
        print("Starting thread: {}".format(self.id))
        with tf.python_io.TFRecordWriter(self.tf_record_path) as writer:
            for i, dir in enumerate(self.dirs):
                img = self.load_image(dir)
                self.save_tf_record(writer, img)
                if i % 1000 == 0:
                    print("Thread {}: {}/{}".format(self.id, i, len(self.dirs)))
            print("Thread {} done".format(self.id))