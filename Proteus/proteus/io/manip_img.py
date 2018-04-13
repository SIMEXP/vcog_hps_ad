__author__ = "Christian Dansereau"
__copyright__ = "Copyright 2015, Christian Dansereau"
from skimage import io
from skimage.filter import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.morphology import label, closing, square
import os
from PIL import Image

#source_path = "/home/cdansereau/git/vision_or/set2/"
#source_path = "/Users/christian/git/vision_or/set2/"
source_path = os.path.abspath("./set2") + "/"
output_path = os.path.abspath("./outset2") + "/"

class img_manager:
    def __init__(self, training_path):
        self.training_path = training_path


        for x in range(10, n_img+10):
            file_name =  "0" + str(x) + ".jpg";
            #print(file_name)
            im = io.imread(source_path + file_name)
            im = im.mean(2)
            if 'img_avg' in locals():
                img_avg = np.concatenate((img_avg,im[:,:,None]),2)
            else:
                img_avg = im[:,:,None]
        return img_avg

def load_image(n_img):
    x = n_img+10
    file_name =  "0" + str(x) + ".jpg";
    im = io.imread(source_path + file_name)
    return im

def show(image):
    plt.imshow(image)
    plt.show()

def showbw(image):
    plt.imshow(image,cmap='Greys')
    plt.show()

def overlay(seg,origin):
    image_label_overlay = label2rgb(label_image, image=imrgb)

def show_overlay(seg,origin):
    out = overlay(seg,origin)
    show(out)

def save(img,item_id,n_image):
    im = Image.fromarray(img)
    file_name =  "0" + str(n_image+10) + '_' + item_id + ".jpg";
    im.save(output_path + file_name)
    

if __name__ == "__main__":
    im = load_image(5).mean(2)
    bw = apply_bw(im)
