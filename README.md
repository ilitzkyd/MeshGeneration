# Mesh Generation and Image Viewing

This repository provides a Python-based framework for generating and visualizing 3D meshes from biological images. The code includes modules for visualizing 3D images, applying growth algorithms, and generating triangle meshes for cell surface models.

## Project Structure

- `Image-Viewer.py`: A script that allows viewing of 3D image data slice by slice with interactive scrolling.
- `Mesh-Test.py`: A script to generate 3D meshes from experimental 3D multichannel images using techniques such as Gaussian filtering and the marching cubes algorithm.
- `Mesh.py`: Contains functions for manipulating and growing 3D binary images, and testing the mesh generation process.

## Getting Started

### Prerequisites

Make sure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `scipy`
- `skimage`
- `meshio`

You can install the necessary dependencies using:

```bash
pip install numpy matplotlib scipy scikit-image meshio

### Running the code

To visualize a 3D image, use the scroll_view function from Image-Viewer.py. You can load a 3D image (in numpy array format) and interactively scroll through the image slices.
Example usage:
from Image-Viewer import scroll_view
import numpy as np

# Load your 3D image as a numpy array
image = np.random.rand(100, 100, 50)

# Visualize the image slice by slice
scroll_view(image)

### Generating Meshes

To generate a mesh from experimental data, use the get_cell_surface function in Mesh-Test.py. This function takes a 3D image and applies the marching cubes algorithm to create a 3D mesh.

from Mesh_Test import get_cell_surface

# Path to the folder containing your .tif images
input_path = "path/to/your/images"
output_mesh = "output_mesh.stl"

# Generate the mesh
get_cell_surface(input_path, output_mesh)



### Testing Mesh Generation
To test mesh generation using sample binary images, use the test function in Mesh.py. It demonstrates applying growth algorithms and comparing pre- and post-growth images.



