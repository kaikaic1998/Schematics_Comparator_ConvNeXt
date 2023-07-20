# ----------------------PDF to Image----------------------
# from pdf2image import convert_from_path

# poppler_path = "./poppler_23_05_0_0/Library/bin"

# dir = './EVT_006/'
# pdf_name = 'EVT_006.pdf'

# # convert PDF to image then to array ready for opencv
# pages = convert_from_path(dir + pdf_name, poppler_path=poppler_path)

# for i , image in enumerate(pages):
#     file_name = 'Page_' + str(i+1) + '.jpg'
#     image.save(dir + file_name, "JPEG")

# #------------------------------------------------

# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

# img = Image.open('./ass_032/Page_1.jpg')
# im = img.resize((224,224))
# im.show()

# #------------------------------------------------
import os

from PIL import Image

import torch
import torchvision.transforms as transform

Dataset_path = './Dataset/'

# loop through the folders in Dataset
for img_folder_name in os.listdir(Dataset_path):
    # loop through the images in all the folders in Dataset
    img_path = os.path.join(Dataset_path, img_folder_name)
    i = 0
    for image_name in os.listdir(img_path):

        i += 1

        orig_img = Image.open(os.path.join(img_path, image_name))

        # -------------Randomly change ColorJitter----------------
        jitter = transform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5), 
                                       saturation=(0.5,1.5), hue=(-0.5,0.5))
        jitted_imgs = [jitter(orig_img) for _ in range(5)]
        
        for j, ColorJiter_img in enumerate(jitted_imgs):
            file_name = 'ColorJitter_' + str(i) + str(j+1) + '.jpg'
            ColorJiter_img.save(img_path + '/' + file_name, 'JPEG')
        #----------------------------------------------------------

        
