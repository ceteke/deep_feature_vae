from vision_project.data_loader import DataLoader
from vision_project.model import Model
from vision_project.utils import build_grid_img
import tensorflow as tf, os, scipy.misc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_name = 'logs/run1'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


sess = tf.Session()

model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

reconstructed_images = model.reconstruct()
grid_img = build_grid_img(reconstructed_images, 64, 64, 10, 10)
im_dir = os.path.join(experiment_name, 'output.jpg')
scipy.misc.imsave(im_dir, grid_img)