from vision_project.model import Model
from vision_project.data_loader import DataLoader
from vision_project.utils import freeze_graph
import tensorflow as tf, os

experiment_name = 'logs/run1'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name, inference=True)

print(model.model_dir)
freeze_graph(sess, '/home/cem/vision_project/logs/run1', "encoder_1/add", "decoder_2/conv2d_3/BiasAdd")