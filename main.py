from vision_project.data_loader import DataLoader
import tensorflow as tf, time

dl = DataLoader('/home/ceteke/Documents/datasets/img_align_celeba', n_dataset=1)
# dl.prepare_images()
#
tic = time.time()
dl.load_dataset()
toc = time.time()
#
dataset = dl.dataset.batch(64)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))

print(toc-tic)