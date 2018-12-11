from vision_project.data_loader import DataLoader
from vision_project.model import Model
import tensorflow as tf, os
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_name = 'logs/run1'

# For training
dl = DataLoader('/home/cem/img_align_celeba')

dataset, num_batches = dl.load_dataset(64)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
model = Model(sess, next_element, '/home/cem/vgg19.npy', experiment_name)

for e in range(10):
    print("Epoch {}".format(e+1))
    sess.run(iterator.initializer)
    for _ in tqdm(range(num_batches)):
        model.fit_batch()

model.save()