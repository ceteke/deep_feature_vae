from vision_project.data_loader import DataLoader
import tensorflow as tf, time

dl = DataLoader('/home/ceteke/Documents/datasets/img_align_celeba')
dl.prepare_images()

tic = time.time()
dl.load_dataset()
toc = time.time()

print(toc-tic)

print(sum(1 for _ in tf.python_io.tf_record_iterator('data.tfrecord')))