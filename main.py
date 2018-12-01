from vision_project.data_loader import DataLoader
from vision_project.model import Model
import tensorflow as tf, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(64)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Model(next_element, '/home/cem/vgg19.npy', num_batches)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(5):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(model.train_op, feed_dict={model.is_train: True})
        except tf.errors.OutOfRangeError:
            break