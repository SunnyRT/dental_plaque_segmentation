
import bpy
import os

# Path to the input and output files
input_filepath = "D:\sunny\Codes\DPS\data_teethseg\origin 101_origin.ply"
output_uv_layout_dir = "D:\sunny\Codes\DPS\data_bpy_map"

# Import the .ply file
bpy.ops.import_mesh.ply(filepath=input_filepath)

# Assume the imported mesh is the active object
obj = bpy.context.active_object

# Ensure we are in edit mode
bpy.ops.object.mode_set(mode='EDIT')

# Select the entire mesh
bpy.ops.mesh.select_all(action='SELECT')

# Perform Smart UV Project to mark seams
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02, user_area_weight=1.0)

# Export UV layout for visualization
image_size = 1024  # Size of the UV layout image
bpy.ops.uv.export_layout(filepath=os.path.join(output_uv_layout_dir, "uv_layout.png"), size=(image_size, image_size), export_all=True)

# Return to object mode to refresh the view
bpy.ops.object.mode_set(mode='OBJECT')
