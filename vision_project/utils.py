import numpy as np

def build_grid_img(inputs, img_height, img_width, n_row, n_col):
    grid_img = np.zeros((img_height*n_row, img_width*n_col, 3))
    print(inputs.shape)
    count = 0
    for i in range(n_col):
        for j in range(n_row):
            grid_img[i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width,:] = inputs[count]
            count += 1
    return grid_img