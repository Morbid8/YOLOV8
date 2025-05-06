import cv2
import os


def check_image_dtypes(folder_path, max_images=100, verbose=True):
    """
    检查指定文件夹内图像的 dtype 和 shape（最多检查 max_images 张）

    :param folder_path: 图像文件夹路径
    :param max_images: 要检查的最大图像数，默认100
    :param verbose: 是否打印输出，默认True
    :return: 包含每张图像信息的列表（dict）
    """
    # 获取图像文件名（按名称排序）
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])[:max_images]

    results = []
    for i, filename in enumerate(image_files):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保留原始深度

        if img is None:
            msg = f"[{i:03}] {filename}: 图像读取失败"
            if verbose:
                print(msg)
            results.append({'index': i, 'filename': filename, 'status': '读取失败', 'dtype': None, 'shape': None})
        else:
            msg = f"[{i:03}] {filename}: dtype = {img.dtype}, shape = {img.shape}"
            if verbose:
                print(msg)
            results.append(
                {'index': i, 'filename': filename, 'status': '读取成功', 'dtype': img.dtype, 'shape': img.shape})

    return results

# folder_0 = r'D:\pythonProject\ultralytics-main\images\images_part1'
# folder_1 = r'D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\gray_images\images_part1'
folder_2 = r'D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\data\images\test'
# results = check_image_dtypes(folder_path=folder_0, max_images=10)
# results = check_image_dtypes(folder_path=folder_1, max_images=10)
results = check_image_dtypes(folder_path=folder_2, max_images=1000)