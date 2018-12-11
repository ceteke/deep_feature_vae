from vision_project.data_loader import DataLoader
from vision_project.model import Model
from vision_project.utils import save_grid_img
import tensorflow as tf, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_name = 'logs/run1'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


sess = tf.Session()

model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name, inference=True)
model.load()

original_image, reconstructed_images = model.reconstruct()

im_dir = os.path.join(experiment_name, 'output.jpg')
save_grid_img(reconstructed_images, im_dir, 64, 64, 10, 10)

org_dir = os.path.join(experiment_name, 'original.jpg')
save_grid_img(original_image, org_dir, 64, 64, 10, 10)

ran_img = model.generate(100)
ran_dir = os.path.join(experiment_name, 'rand.jpg')
save_grid_img(ran_img, ran_dir, 64, 64, 10, 10)

