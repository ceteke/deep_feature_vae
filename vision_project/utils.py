import numpy as np, scipy.misc, cv2, tensorflow as tf
from PIL import Image

def build_grid_img(inputs, img_height, img_width, n_row, n_col):
    grid_img = np.zeros((img_height*n_row, img_width*n_col, 3))
    count = 0
    for i in range(n_row):
        for j in range(n_col):
            grid_img[i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width,:] = inputs[count]
            count += 1
    return grid_img

def register_extension(id, extension):
    Image.EXTENSION[extension.lower()] = id.upper()

def register_extensions(id, extensions):
    for extension in extensions: register_extension(id, extension)

def save_grid_img(inputs, path, img_height, img_width, n_row, n_col):
    Image.register_extension = register_extension
    Image.register_extensions = register_extensions
    grid_img = build_grid_img(inputs, img_height, img_width, n_row, n_col)
    scipy.misc.imsave(path, grid_img)

def load_image(img_dir):
    return cv2.imread(img_dir)


def freeze_graph(sess, model_dir, *output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    clear_devices = True

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    saver.restore(sess, input_checkpoint)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        list(output_node_names)
    )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def