from vision_project.data_loader import DataLoader
from vision_project.model import Model
from vision_project.utils import save_grid_img
import tensorflow as tf, os, numpy as np, pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_name = 'logs/run1'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(32)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()

model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name, inference=True)
model.load()

file_dir = os.path.join(experiment_name, 'Eyeglasses_vec.pk')
vector = pickle.load(open(file_dir, 'rb'))

changes = []
batch = sess.run(next_element)
img = batch[0]

latent = model.encode(img)

for i in range(10):
    alpha = i/float(9)
    latent_int = latent + alpha*vector*2
    changes.append(model.decode(latent_int))

grid_path = os.path.join(experiment_name, 'attribute.jpg')
save_grid_img(changes, grid_path, 64, 64, 1, 10)