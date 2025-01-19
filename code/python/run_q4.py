import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap="Greys")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()

    # Sort boxes by top edge (y1)
    sorted_boxes = sorted(bboxes, key=lambda box: box[0])

    lines = []
    current_line = [sorted_boxes[0]]
    line_height = sorted_boxes[0][2] - sorted_boxes[0][0]  # Height of the first box

    for box in sorted_boxes[1:]:
        # Check if the box belongs to the current line
        if (
            abs(box[0] - current_line[-1][0]) < line_height * 0.5
        ):  # Adjust threshold as needed
            current_line.append(box)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [box]

        # Update line height (average of boxes in the line)
        line_height = np.mean([b[2] - b[0] for b in current_line])

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Sort boxes within each line by left edge (x1)
    sorted_lines = [sorted(line, key=lambda box: box[1]) for line in lines]

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    # create a vector to store cropped, resized, and flattened letters given by each bounding box
    crops = []
    for line in sorted_lines:
        for box in line:
            y1, x1, y2, x2 = box  # current letter bounding box
            crop = bw[y1:y2, x1:x2]  # crop the letter (only the part inside the bounding box)

            # calculate padding to make it square
            height, width = crop.shape
            size_diff = abs(height - width)
            pad_top = pad_bottom = pad_left = pad_right = 0

            if height > width:  # taller than wide
                pad_left = size_diff // 2
                pad_right = size_diff - pad_left
            elif width > height:  # wider than tall
                pad_top = size_diff // 2
                pad_bottom = size_diff - pad_top

            # invert the crop
            crop = 1 - crop

            # # show the crop
            # plt.imshow(crop, cmap="Greys")
            # plt.show()

            # pad the crop to make it square
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=1)

            # dialate the crop to make lines thicker
            d = np.ones((int(11*crop.shape[0]/crop.shape[1]), int(10*crop.shape[1]/crop.shape[0]))) # for letter dilation
            crop = skimage.morphology.erosion(crop, d)
            # crop = skimage.morphology.erosion(crop, np.ones((3, 3)))
            #crop = skimage.morphology.dilation(crop, np.ones((6, 6)))

            # resize the crop to 32x32
            crop = skimage.transform.resize(
                crop, (32, 32), anti_aliasing=True, preserve_range=True
            )

            # show the resized crop
            # plt.imshow(crop, cmap="Greys")
            # plt.show()

            # traspose the crop
            crop = crop.T

            # flatten the crop
            crops.append(crop.flatten())

    inputs = np.array(crops)

    print("Found {} letters".format(inputs.shape[0]))

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))

    # forward pass
    h1 = forward(inputs, params, "layer1")
    probs = forward(h1, params, "output", softmax)

    # get the most likely letter for each crop
    most_likely = np.argmax(probs, axis=1)
    most_likely_letters = "".join(letters[most_likely])

    # separate the letters by line
    start = 0
    for line in sorted_lines:
        end = start + len(line)
        print(most_likely_letters[start:end])
        start = end
    print("\n")
    