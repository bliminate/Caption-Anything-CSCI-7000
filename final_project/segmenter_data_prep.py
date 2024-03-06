#!/usr/bin/python3

from PIL import Image
import random
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import math

# Pick random points in image to segment, and crop

class ImageFilenameStruct:
    def __init__(self, filepath, name, name_wo_ext):
        self.filepath = filepath
        self.name = name
        self.name_wo_ext = name_wo_ext

def loadSegmenter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if 'cuda' in device else torch.float32
    model_type = 'vit_h'
    checkpoint = '../model_checkpoints/sam_vit_h_4b8939.pth'

    # SAM initialization
    model = sam_model_registry[model_type](checkpoint = checkpoint)
    model.to(device)
    predictor = SamPredictor(model)
    #mask_generator = SamAutomaticMaskGenerator(model)

    return predictor

def loadImageFilepaths():
    # load image filepaths
    images = []
    with open("images.txt", "r") as imageFiles:
        for line in imageFiles.readlines():
            line = line.strip()
            filename = line.split("/")[-1]
            filenameNoExt = filename.split(".")[0]
            images.append(ImageFilenameStruct(line, filename, filenameNoExt))
    return images

def parseAndCropImages(predictor, imageFiles):

    count = 1
    for imageFile in imageFiles:
        print("Processing image " + str(count) + "...")
        image = Image.open(imageFile.filepath)
        width, height = image.size
        predictor.set_image(np.array(image)) # load the image to predictor

        for i in range(0,5):
            inx = math.floor(width/2)
            iny = math.floor(height/2)

            if i == 0:
                # horiz, centered
                # vert 1/4 from top 
                iny = math.floor(height/4)
            elif i == 1:
                # vert centered
                # horiz 1/4 from left
                inx = math.floor(width/4)
            elif i == 2:
                pass # centered
            elif i == 3:
                # vert centered
                # horiz 1/4 from right
                inx = math.floor(width*(3/4))
            elif i == 4:
                # horiz centered
                # vert 1/4 from bottom
                iny = math.floor(height*(3/4))

            input_point = np.array([[inx, iny]])
            input_label = np.array([1])

            masks, scores, logits = predictor.predict(point_coords = input_point, point_labels = input_label)
            masks = masks[0, ...]
            bitmaskImage = Image.fromarray(masks)
            save_path = "../data/" + imageFile.name_wo_ext + "_" + str(i) + "_bitmask.jpg"
            save_masks = "../data/" + imageFile.name_wo_ext + "_" + str(i) + "_mask.txt"
            bitmaskImage.save(save_path)
            with open(save_masks, "w") as maskOut:
                maskOut.writelines(np.array2string(masks))
        
        count += 1
    
    return

if __name__ == "__main__":
    print("Loading image filepaths...")
    images = loadImageFilepaths()
    print("Loading segmenter...")
    predictor = loadSegmenter()
    print("Generating bitmasks...")
    parseAndCropImages(predictor, images)
