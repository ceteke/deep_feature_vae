from vision_project.data_loader import AttributeLoader
from vision_project.model import Model
import tensorflow as tf, os, numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

attribute_name = 'Mustache'
attribute_latent = []
attribute_label = []

data_dir = '/home/cem/img_align_celeba'
experiment_name = 'logs/run1'

al = AttributeLoader('/home/cem/list_attr_celeba.csv')
next_element, num_batch, n_example = al.get_ids_iterator(data_dir, attribute_name)

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

print(attribute_name)
attribute_label.append([1]*n_example)
for _ in tqdm(range(num_batch)):
    attribute_latent.append(model.get_latent())

######
tf.reset_default_graph()

al = AttributeLoader('/home/cem/list_attr_celeba.csv')
next_element, num_batch, n_example = al.get_ids_iterator(data_dir, '~'+attribute_name, n=10000)

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)
model.load()

print('~'+attribute_name)
attribute_label.append([0]*n_example)
for _ in tqdm(range(num_batch)):
    attribute_latent.append(model.get_latent())

attribute_latent = np.concatenate(attribute_latent)
attribute_label = np.concatenate(attribute_label)

print(attribute_latent.shape, attribute_label.shape)

# SVM

train_latent, test_latent, train_label, test_label = train_test_split(attribute_latent, attribute_label, test_size=0.1)

print("Training Linear SVM")
svm = LinearSVC()
svm.fit(train_latent, train_label)
acc = svm.score(test_latent, test_label)
print("Accuracy for {} is {}".format(attribute_name, acc))