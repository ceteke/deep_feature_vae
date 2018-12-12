from vision_project.model import Model
import os, tensorflow as tf, numpy as np
from vision_project.data_loader import DataLoader
from vision_project.utils import save_grid_img

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_name = 'logs/run1'

dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(32)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

batch = sess.run(next_element)

imgs = []
for i in range(6):
    img1 = batch[i]
    img2 = batch[i+1]

    imgs.append(model.interpolate(img1, img2))

imgs = np.concatenate(imgs)
print(imgs.shape)

int_path = os.path.join(experiment_name, 'interpolate.jpg')
save_grid_img(imgs, int_path, 64, 64, 6, 12)