# split_lungs.py

import pyvista as pv
import numpy as np

mesh = pv.read("lung_model/lungs.stl")

center_x = mesh.center[0]

left = mesh.clip(
    normal=(1,0,0),
    origin=(center_x,0,0),
    invert=True
)

right = mesh.clip(
    normal=(1,0,0),
    origin=(center_x,0,0)
)

left.save("lung_model/left_lung.stl")
right.save("lung_model/right_lung.stl")

print("DONE")