import os
import pathlib
import time
import datetime
import random
import string
import numpy as np
import cv2
from skimage import exposure 
from skimage.filters import unsharp_mask

# 图片灰度转换
# 设置输入和输出目录
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
inputdirs = [
    r"D:\pythonProject\ultralytics-main\images\images_part1",
    r"D:\pythonProject\ultralytics-main\images\images_part2",
    r"D:\pythonProject\ultralytics-main\images\images_part3",
    r"D:\pythonProject\ultralytics-main\images\images_part4"
]
outputdir = pathlib.Path(os.path.join(dataset_root, "gray_images"))  # 输出目录：gray_images

# 设置选项
filetypes = set(['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'])  # 支持的输入文件类型
output_filetype = '.png'  # 输出文件类型
jpg_compression = '30'  # JPG 压缩质量（仅对 JPG 输出有效）
outputbitdepth = 8  # 输出位深度（8 或 16 位）
sharpen = False  # 是否锐化
convert_grayscale = True  # 是否转换为灰度图
equalize = True  # 是否应用 CLAHE 对比度增强
intensity_crop = 0.05  # 强度裁剪百分比（用于 CLAHE）
filename_random = False  # 是否生成随机文件名
relative = True  # 是否保留文件夹结构
overwrite = False  # 是否覆盖已有文件

# 生成随机字符串（用于替换文件名）
def get_random_alphaNumeric_string(stringLength=8):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))

# 定义主函数
def main():
    for inputdir in inputdirs:
        inputdir = pathlib.Path(inputdir)
        # 检查输入目录是否存在
        if not os.path.exists(inputdir):
            print(f"Error (Directory error): Input directory does not exist: {inputdir}")
            continue

        # 遍历每个输入目录
        for paths, _, files in os.walk(os.path.normpath(inputdir), topdown=True):
            for i, file in enumerate(files, start=1):
                time_index = str(time.strftime("%H:%M:%S", time.localtime())) + ' (' + str(i).zfill(8) + ')'
                try:
                    filepath = os.path.join(paths, file)
                    reldir = os.path.relpath(paths, os.path.dirname(inputdir))  # 相对于 images 目录
                    if relative:
                        outputpath = os.path.normpath(os.path.join(outputdir, reldir))
                    else:
                        outputpath = os.path.normpath(outputdir)

                    if filename_random:
                        filename = get_random_alphaNumeric_string(32)
                    else:
                        filename = file

                    if '.' not in output_filetype:
                        output_filetype_corr = '.' + output_filetype
                    else:
                        output_filetype_corr = output_filetype

                    outputfile = os.path.normpath(pathlib.Path(os.path.join(outputpath, filename)).with_suffix(output_filetype_corr))
                    if overwrite == False and os.path.isfile(outputfile):
                        print(f'SKIPPED (File exists), {filepath} - {time_index}')
                        continue
                    else:
                        if any(x in filepath.lower() for x in filetypes):
                            if not os.path.exists(outputpath):
                                os.makedirs(outputpath)
                            if convert_grayscale:
                                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                            else:
                                img = cv2.imread(filepath, -cv2.IMREAD_ANYDEPTH)

                            if img is None:
                                print(f'ERROR (Failed to read image), {filepath} - {time_index}')
                                continue

                            if img.dtype == 'uint16':
                                img = (img / 65535.0).astype(np.float64)
                            elif img.dtype == 'uint8':
                                img = (img / 255.0).astype(np.float64)
                            else:
                                print(f'ERROR (Input bit depth not supported), {filepath} - {time_index}')
                                continue

                            if equalize:
                                img = exposure.rescale_intensity(img, in_range=(np.percentile(img, intensity_crop), np.percentile(img, (100-intensity_crop))))
                                img = exposure.equalize_adapthist(img)
                            if sharpen:
                                img = unsharp_mask(img, radius=1, amount=1)

                            if outputbitdepth == 8:
                                img = cv2.normalize(img, dst=None, alpha=0, beta=int((pow(2, outputbitdepth))-1), norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                            elif outputbitdepth == 16:
                                img = cv2.normalize(img, dst=None, alpha=0, beta=int((pow(2, outputbitdepth))-1), norm_type=cv2.NORM_MINMAX).astype(np.uint16)
                            else:
                                print(f'ERROR (Output bit depth not supported), {filepath} - {time_index}')
                                continue

                            if ('jpg' in output_filetype.lower() or 'jpeg' in output_filetype.lower()) and jpg_compression:
                                cv2.imwrite(outputfile, img, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_compression)])
                                print(f'SUCCESS (Conversion succeeded), {filepath} - {time_index}')
                            else:
                                cv2.imwrite(outputfile, img)
                                print(f'SUCCESS (Conversion succeeded), {filepath} - {time_index}')
                except Exception as e:
                    print(f'ERROR (Conversion failed), {filepath} - {time_index}, Exception: {str(e)}')
        else:
            pass

# 调用主函数
if __name__ == "__main__":
    main()