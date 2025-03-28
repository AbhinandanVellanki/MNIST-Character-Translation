import numpy as np

import matplotlib.pyplot as plt

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # denoise
    image_denoised = skimage.restoration.denoise_bilateral(image, channel_axis=-1)

    # greyscale
    image_gray = skimage.color.rgb2gray(image_denoised)

    # show image
    # skimage.io.imshow(image_gray)
    # skimage.io.show()

    # threshold
    b_w_threshold = skimage.filters.threshold_otsu(image_gray)

    # morphology
    bw = image_gray < b_w_threshold
    bw = skimage.morphology.binary_closing(bw, skimage.morphology.square(3))
    dilated_bw = skimage.morphology.dilation(bw, np.ones((9, 9))) # dilate to connect strokes into letters
    # show image
    # skimage.io.imshow(dilated_bw)
    # skimage.io.show()

    # label
    label_image = skimage.measure.label(skimage.segmentation.clear_border(dilated_bw))
    regions = skimage.measure.regionprops(label_image)

    # small region threshold
    small_region_threshold = 300

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= small_region_threshold:
            # draw rectangle
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minr, minc, maxr, maxc])

    return bboxes, bw
