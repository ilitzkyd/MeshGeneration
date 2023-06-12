import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage import data
from skimage.segmentation import find_boundaries
from skimage.draw import line_nd

from skimage.color import label2rgb
from scipy.spatial.distance import cdist
from util_image_viewer import scatter_bw_img, scroll_view, scroll_view_compare


def test():
    # Load binary image
    bw_img = np.load("test_data/20deg_ellipses_2.npy")

    # Apply growth algorithm
    bw_img_after_growth = growth_v1(bw_img)

    # Compare the image before and after growth
    scatter_bw_img(bw_img)
    scatter_bw_img(bw_img_after_growth)
    scroll_view_compare(bw_img, bw_img_after_growth)


def growth_v1(bw_img):
    img_after_growth = np.copy(bw_img)

    # Label raw image
    label_img = label(bw_img, connectivity=bw_img.ndim)
    props = regionprops(label_img)
    assert len(props) > 0, "No connected volumes found."

    nrows, ncols, nz = bw_img.shape
    QUERY = False

    if QUERY:
        print(props[0].label)
        print(props[0].bbox)
        print(props[0].image.shape)
        print(find_boundaries(props[0].image).shape)

    def find_region_boundaries(region):
        bbox = region.bbox
        boundary_coords = np.array(np.where(find_boundaries(region.image)))
        boundary_coords[0, :] += bbox[0]
        boundary_coords[1, :] += bbox[1]
        boundary_coords[2, :] += bbox[2]
        boundary_coords = np.transpose(boundary_coords)
        return boundary_coords

    def connect(region1, region2, img):
        X1 = find_region_boundaries(region1)
        X2 = find_region_boundaries(region2)
        Y = cdist(X1, X2)

        # For each boundary point of region2, find the nearest point on the boundary of
        # region1, then connect the two points
        coords = np.where(Y == np.amin(Y, axis=0))
        for n1, n2 in list(zip(coords[0], coords[1])):
            p1 = X1[n1, :]
            p2 = X2[n2, :]
            lin = line_nd(p1, p2)
            img[lin] = 1

    connect(props[0], props[1], img_after_growth)
    connect(props[1], props[0], img_after_growth)
    img_after_growth = ndimage.morphology.binary_fill_holes(img_after_growth)

    # Use bounding rectangles to mark different regions
    fig, ax = plt.subplots(1, 1)
    for region in props:
        minr, minc, minh, maxr, maxc, maxh = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.tight_layout()

    # View the original label maps
    scroll_view(label_img, fig, ax)
    plt.show()

    return img_after_growth


def growth_v0(bw_img):
    img_after_growth = np.copy(bw_img)

    # Label raw image
    label_img = label(bw_img, connectivity=bw_img.ndim)
    props = regionprops(label_img)
    assert len(props) > 0, "No connected volumes found."

    nrows, ncols, nz = bw_img.shape
    D = np.sqrt(nz * nz + nrows * nrows + ncols * ncols)

    def score(region, shape):
        score_map = np.zeros(shape)
        centroid = region.centroid
        for index, x in np.ndenumerate(score_map):
            d = np.linalg.norm(np.array(index) - np.array(centroid))
            score_map[index] = np.exp(-d / D)
        return score_map

    score_map_a = score(props[0], bw_img.shape)
    score_map_b = score(props[1], bw_img.shape)
    score_map_sum = score_map_a + score_map_b
    for index, x in np.ndenumerate(score_map_sum):
        if x > 1.6:
            img_after_growth[index] = 1
        else:
            img_after_growth[index] = 0

    # Use bounding rectangles to mark different regions
    fig, ax = plt.subplots(1, 1)
    for region in props:
        minr, minc, minh, maxr, maxc, maxh = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.tight_layout()

    # View the original label maps
    scroll_view(label_img, fig, ax)
    plt.show()

    return img_after_growth


if __name__ == "__main__":
    # Comment: There is an inconsistency among the coordinate conventions
    # of the code with skimage. For 3D grayscale images, the order convention in skimage is (pln, row, col).
    # For convenience, the current code adopts the order (row, col, pln).
    test()
