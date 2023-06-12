import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image
import time
import meshio
from scipy import ndimage
from skimage.measure import label, regionprops, marching_cubes_lewiner
from skimage.external import tifffile


def scatter_bw_img(bw_img, x_resolution, y_resolution, z_resolution, max_dots=7000):
    """
    Visualize a 3D image. Used for testing the data at various stages in the pipeline
    :param bw_img: np array - binary 3D image
    :param x_resolution:
    :param y_resolution:
    :param z_resolution:
    :param max_dots:
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    indices_pix = np.argwhere(bw_img)
    indices = indices_pix * np.array([x_resolution, y_resolution, z_resolution])
    n = indices.shape[0]
    delta = int(n / max_dots)

    # Hacky-axis-equal taken from https://stackoverflow.com/a/13701747/11756605
    max_range = np.array([indices[:, 0].max() - indices[:, 0].min(),
                          indices[:, 1].max() - indices[:, 1].min(),
                          indices[:, 2].max() - indices[:, 2].min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (indices[:, 0].max() + indices[:, 0].min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (indices[:, 1].max() + indices[:, 1].min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (indices[:, 2].max() + indices[:, 2].min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.scatter(indices[::delta, 0],
               indices[::delta, 1],
               indices[::delta, 2],
               s=2,
               c='r')
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    plt.show()


def tif_reader(path, color_idx=0):
    """
    Read a 3D image stored as .tif files in a folder into a np array
    :param path:
    :param color_idx:
    :return: np array - 3D image
    :return: float - xy_resolution (um/pix)
    """

    fnames = glob.glob(path + '/*.tif')
    fnames.sort()
    num_images = len(fnames)
    sample_img = plt.imread(fnames[0])

    size_x = sample_img.shape[0]
    size_y = sample_img.shape[1]

    # Read resolution from .tif metadata
    with tifffile.TiffFile(fnames[0]) as tif:
        resolution_tuple = tif.pages[0].tags['x_resolution'].value

    all_array = np.zeros((size_x, size_y, num_images))

    for idx, fname in enumerate(fnames):
        img = plt.imread(fname)
        all_array[:, :, idx] = img[:, :, color_idx]

    return all_array, resolution_tuple[1]/resolution_tuple[0]


def get_cell_surface(path, output_path, z_resolution=0.8, ss=1):
    """
    Takes the raw experimental data (3D multichannel image) and generates a raw triangle mesh

    :param path:
    :param output_path:
    :param z_resolution:
    :param ss: int - step size for marching_cubes method
    :return:
    """

    intensity_threshold = 1
    area_threshold = 40

    # import the image file
    raw_img, xy_resolution = tif_reader(path)
    # apply Gaussian filter
    filtered_img = ndimage.gaussian_filter(raw_img, 1)
    # threshold the image
    bw_img = filtered_img > intensity_threshold

    # find connected volumes
    label_img = label(bw_img, connectivity=bw_img.ndim)
    print("Labeled")
    props = regionprops(label_img)
    print("Propped")
    assert len(props) > 0, "No connected volumes found."
    areas = []
    for idx, p in enumerate(props):
        print(str(idx) + "/" + str(len(props)))
        areas.append(p.area)
        if p.area < area_threshold:
            bw_img[label_img == (idx+1)] = 0
    scatter_bw_img(bw_img, xy_resolution, xy_resolution, z_resolution, max_dots=12000)
    arg = np.argmax(areas)

    # isolate the cell
    bw_img_cell = np.zeros(bw_img.shape)
    cell_indices = np.argwhere(label_img == (arg + 1))
    bw_img_cell[cell_indices[:, 0],
                cell_indices[:, 1],
                cell_indices[:, 2]] = 1

    # flip the ii and jj dimensions to be compatible with the marching cubes algorithm
    bw_img_cell = np.swapaxes(bw_img_cell, 0, 1)

    # get the cell surface mesh from the marching cubes algorithm and the isolated cell image
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, normals, _ = marching_cubes_lewiner(bw_img_cell,
                                                      spacing=(xy_resolution,
                                                               xy_resolution,
                                                               z_resolution),
                                                      step_size=ss)

    # save surface mesh info
    meshio.write(output_path + "_ss" + str(ss) + ".stl",
                 meshio.Mesh(points=verts,
                             cells={"triangle": faces}))


if __name__ == "__main__":

    path_to_raw_images = "Cell 2"  # Go all the way to '.../Cell'
    output_path = "mesh_gen.stl"

    get_cell_surface(path_to_raw_images,
                     output_path,
                     ss=1)
