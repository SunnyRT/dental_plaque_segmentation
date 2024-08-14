import open3d as o3d
import numpy as np
import os,sys

path = r"D:\sunny\Codes\DPS\data_raw_teethseg\origin"
all_files = os.listdir(path)

save_path = r"D:\sunny\Codes\DPS\data_8w\3d_ply\origin"

for file in all_files:
    if file == '000702.ply' or file == '000801.ply':
        filepath = os.path.join(path,file)
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=80000)
        o3d.io.write_triangle_mesh(os.path.join(save_path, file), mesh_smp)
        print(file)

print("Done")




# for i in range(2,6):
#     for file in dirs:
#         subpath = os.path.join(path,file)

#         # ply_path = os.path.join(subpath,ply)
#         mesh = o3d.io.read_triangle_mesh(subpath)
#         # mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
#         plyname = file+ "_"+str(i)+".ply"
        
#         o3d.io.write_triangle_mesh(plyname, mesh)
#         num += 1 
#         print(plyname)
    # else:
    #     mesh = o3d.io.read_triangle_mesh(subpath)
    #     mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    #     # plyname = str(num).zfill(4) + ".ply"
    #     plyname = str(onum).zfill(4) + "_origin.ply"
    #     o3d.io.write_triangle_mesh(plyname, mesh_smp)
    #     onum+=1
    #     ply_path = os.path.join(path,ply)
    #     mesh = o3d.io.read_triangle_mesh(ply_path)
    #     mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=16000)
    #     plyname = str(onum).zfill(4) + "_origin.ply"
    #     o3d.io.write_triangle_mesh(plyname, mesh_smp)
    #     onum += 1 
    #     print(plyname)


# path = r"OneDrive_3_11-30-2023 13-20 ori"
# dirs = os.listdir(path)

# num = 34
# onum =34
# for file in dirs:
#     # print(ply[3])
#     # print(file)
#     subpath = os.path.join(path,file)
#     subdir = os.listdir(subpath)
#     # # print(subd)
#     for ply in subdir:
#         print(ply)
#     # # if ply[3] == "L":
#         ply_path = os.path.join(subpath,ply)
#         mesh = o3d.io.read_triangle_mesh(ply_path)
#         mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=16000)
#         plyname = str(num).zfill(4) + "_origin.ply"
#         o3d.io.write_triangle_mesh(plyname, mesh_smp)
#         num += 1 
#         print(plyname)


    # else:
    #     ply_path = os.path.join(path,ply)
    #     mesh = o3d.io.read_triangle_mesh(ply_path)
    #     mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=16000)
    #     plyname = str(onum).zfill(4) + "_origin.ply"
    #     o3d.io.write_triangle_mesh(plyname, mesh_smp)
    #     onum += 1 
    #     print(plyname)