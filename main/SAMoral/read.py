import numpy as np

# 加载 .npz 文件（推荐使用 with 语句自动关闭文件）
# path = '/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/train/label_mesh/000102.npz'
path = '/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/train/SOTA_mesh/000102.npz'

with np.load(path, allow_pickle=True) as data:
    # 查看所有存储的数组名称（键）
    print("文件中的数组键：", data.files)
    vertices = data['up']
    print("\n顶点坐标（vertices）形状：", vertices[0]) #(6,)
    # print("\n顶点坐标（vertices）形状：", vertices[5].shape) #(6,)
    # print("\n顶点坐标（vertices）形状：", vertices[0][123])