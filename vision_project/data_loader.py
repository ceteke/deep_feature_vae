import os, tensorflow as tf, math, pandas as pd

class AttributeLoader(object):
    def __init__(self, dir):
        self.dir = dir
        self.attribute_df = pd.read_csv(dir)

    def get_ids(self, attribute):
        if attribute[0] == '~':
            val = -1
            attribute = attribute[1:]
        else:
            val = 1
        return self.attribute_df.loc[self.attribute_df[attribute] == val]['image_id'].values

    def get_ids_iterator(self, data_dir, attribute, n=1000):
        imgs = self.get_ids(attribute)

        eyeglass_dl = DataLoader(data_dir, imgs)

        dataset, num_batches = eyeglass_dl.load_dataset(n)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next(), num_batches, len(imgs)

class DataLoader(object):
    def __init__(self, base_dir, filenames=None):
        self.base_dir = base_dir
        self.filenames = filenames

    def crop_center_image(self, img):
        width_start = tf.cast(tf.shape(img)[1] / 2 - 150 / 2, tf.int32)
        height_start = tf.cast(tf.shape(img)[0] / 2 - 150 / 2, tf.int32)
        cropped_img = img[height_start: height_start + 150, width_start: width_start + 150, :]
        return cropped_img

    def load_dataset(self, batch_size):
        if self.filenames is None:
            filenames = os.listdir(self.base_dir)
        else:
            filenames = self.filenames
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
        image = self.crop_center_image(image)
        image = tf.image.resize_images(image, [64, 64])
        return image

    def train_preprocess(self, image):
        return image