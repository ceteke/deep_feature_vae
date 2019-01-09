from scipy import misc

def crop_center_image(img):
    width_start = int(img.shape[1]/2 - 225/2)
    height_start = int(img.shape[0]/2 - 225/2)
    cropped_img = img[height_start: height_start+260, width_start: width_start+250, :]
    return cropped_img

def load_img(dir):
    image = misc.imread(dir)
    image = image / 255.
    #image = crop_center_image(image)
    return image