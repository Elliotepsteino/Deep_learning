import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
# Root directory of the project
video =0
ROOT_DIR = os.path.abspath("../../")
Sample_dir = os.path.join(ROOT_DIR, "samples")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(Sample_dir)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from samples import pears
from samples.pears import pear
from mrcnn import visualize as viz

#class InferenceConfig(pear.BalloonConfig()):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
 #   GPU_COUNT = 1
 #   IMAGES_PER_GPU = 1

weights_path = "C:\\Users\\ellio\\PycharmProjects\\Mask_RCNN\\logs\\mask_rcnn_pear_0030.h5"
image = "C:\\Users\\ellio\\PycharmProjects\\Mask_RCNN\\datasets\\pear_upd\\test_set\\Conference-Pear-Tree.jpg"

config = pear.BalloonConfig()
config.display()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference",model_dir = MODEL_DIR, config=config)
model.load_weights(weights_path, by_name=True)
image = skimage.io.imread(image)
r = model.detect([image], verbose=1)[0]
viz.display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'],
                                title="Predictions")
#def detect_and_color_splash(model, image_path=None, video_path=None,image):
##    assert image_path or video_path#
#
#    # Image or video?
#    if image_path:
#        # Run model detection and generate the color splash effect
##        print("Running on {}".format(image))
 #       # Read image
 #       image = skimage.io.imread(image)
  #      # Detect objects
  #      r = model.detect([image], verbose=1)[0]
  #      # Color splash
 #       splash = color_splash(image, r['masks'])
 #       # Save output
 #       file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
 #       skimage.io.imsave(file_name, splash)
#
#    print("Saved to ", file_name)
#pear.detect_and_color_splash(model, image_path=image)
