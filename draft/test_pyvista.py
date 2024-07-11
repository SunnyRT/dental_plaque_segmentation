import numpy as np
import cv2
import pyvista as pv

# Create a simple plot
sphere = pv.Sphere()
plotter = pv.Plotter()
plotter.add_mesh(sphere)
plotter.show()

print(f"PyVista version: {pv.__version__}")
print(f"VTK version: {pv._vtk.vtkVersion.GetVTKVersion()}")