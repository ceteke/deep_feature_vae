import numpy as np, scipy.misc
from PIL import Image

def build_grid_img(inputs, img_height, img_width, n_row, n_col):
    grid_img = np.zeros((img_height*n_row, img_width*n_col, 3))
    count = 0
    for i in range(n_col):
        for j in range(n_row):
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
