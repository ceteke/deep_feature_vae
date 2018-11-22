from vision_project.data_loader import DataLoader
import tensorflow as tf, time

dl = DataLoader('/home/ceteke/Documents/datasets/img_align_celeba')

dataset = dl.load_dataset(64)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()

for e in range(100):
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_element).shape)
        except tf.errors.OutOfRangeError:
            break