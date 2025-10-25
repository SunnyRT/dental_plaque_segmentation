import numpy as np
import matplotlib.pyplot as plt
import random


# Get bounding boxes which contain the teeth area
def get_bbox_teeth(origin_patch):
    # Assume origin_patch is of 3 rgb color channel
    # Find non-black pixels
    non_black_pixels = np.any(origin_patch != [0, 0, 0], axis=-1)

    # Get the coordinates of non-black pixels
    non_black_indices = np.argwhere(non_black_pixels)

    # Get the bounding box coordinates
    x_min, y_min = non_black_indices.min(axis=0)
    x_max, y_max = non_black_indices.max(axis=0)

    bbox = [x_min, y_min, x_max, y_max]

    return bbox   

def find_regions(mask):
        regions = []
        for i in range(mask.size(1)):
            for j in range(mask.size(2)):
                if mask[0][i][j] == 1:
                    res = [i, j,i, j]
                    mask[0][i][j] = 0
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if x > 0 and mask[0][x-1][y] == 1:
                            mask[0][x-1][y] = 0
                            res[0] = min(res[0], x-1)
                            stack.append((x-1, y))
                        if x < mask.size(1)-1 and mask[0][x+1][y] == 1:
                            mask[0][x+1][y] = 0
                            res[2] = max(res[2], x+1)
                            stack.append((x+1, y))
                        if y > 0 and mask[0][x][y-1] == 1:
                            mask[0][x][y-1] = 0
                            res[1] = min(res[1], y-1)
                            stack.append((x, y-1))
                        if y < mask.size(2)-1 and mask[0][x][y+1] == 1:
                            mask[0][x][y+1] = 0
                            res[3] = max(res[3], y+1)
                            stack.append((x, y+1))
                    regions.append(res)
        return regions



def visualize_sample(images, masks):
    img_num = random.randint(0, images.shape[0] - 1)
    example_image = images[img_num]
    example_mask = masks[img_num]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(example_image)
    axes[0].set_title("Image")

    axes[1].imshow(example_mask, cmap='gray')
    axes[1].set_title("Mask")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()