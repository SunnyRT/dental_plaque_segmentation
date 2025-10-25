import cv2
import numpy as np

# 图片路径
img_path = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/2Dpred/001302_0.png"

# 1. 读取图片（OpenCV默认读取为BGR格式，需转换为RGB）
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败，请检查路径是否正确！")
    exit()

# 转换为RGB格式（消除OpenCV的BGR通道顺序影响）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 处理可能的Alpha通道（若图片为RGBA格式）
if img_rgb.shape[-1] == 4:
    img_rgb = img_rgb[:, :, :3]  # 只保留RGB通道

# 3. 将图片转换为像素列表（形状为：(像素总数, 3)）
pixels = img_rgb.reshape(-1, 3)  # 展平为二维数组，每行代表一个像素的RGB值

# 4. 定义纯黑和纯白的RGB值
black = np.array([0, 0, 0])
white = np.array([255, 255, 255])

# 5. 检查每个像素是否为纯黑或纯白
#   - 对每个像素，判断是否等于纯黑或纯白（三通道全匹配）
is_black_or_white = np.logical_or(
    np.all(pixels == black, axis=1),  # 所有通道等于纯黑
    np.all(pixels == white, axis=1)   # 所有通道等于纯白
)

# 6. 统计不符合条件的像素
invalid_pixels = pixels[~is_black_or_white]  # 筛选出非纯黑/纯白的像素
invalid_count = len(invalid_pixels)

# 7. 输出结果
if invalid_count == 0:
    print("图片中所有像素均为纯黑或纯白！")
else:
    print(f"图片中存在 {invalid_count} 个非纯黑/纯白的像素，部分像素值如下：")
    # 打印前10个无效像素（避免输出过多）
    for p in invalid_pixels[:10]:
        print(f"像素值：{p}")