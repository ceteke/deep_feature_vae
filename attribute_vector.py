from vision_project.data_loader import AttributeLoader
from vision_project.model import Model
import tensorflow as tf, os, numpy as np, pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

attribute_name = 'Eyeglasses'

data_dir = '/home/cem/img_align_celeba'
experiment_name = 'logs/run1'

al = AttributeLoader('/home/cem/list_attr_celeba.csv')
next_element = al.get_ids_iterator(data_dir, attribute_name)

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

attribute_latent = model.get_latent()
attribute_latent = np.mean(attribute_latent, axis=0)

######
tf.reset_default_graph()

al = AttributeLoader('/home/cem/list_attr_celeba.csv')
next_element = al.get_ids_iterator(data_dir, '~'+attribute_name)

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

no_attribute_latent = model.get_latent()
no_attribute_latent = np.mean(no_attribute_latent, axis=0)

attribute_vector = attribute_latent - no_attribute_latent

file_name = attribute_name + '_vec.pk'
file_path = os.path.join(experiment_name, file_name)
pickle.dump(attribute_vector, open(file_path, 'wb'))