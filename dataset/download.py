import os
import urllib.request
import csv
from tqdm import tqdm

imgPath = """https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fs8.sinaimg.cn%2Fbmiddle%2F93046cecnb0bd2cf8c6b7%26690&refer=http%3A%2F%2Fs8.sinaimg.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1632599062&t=8d3ad00c13dc68f1093280c8f0aaf524"""

def get_response(imagePath, url):
    res = urllib.request.urlopen(url)
    data = res.read()
    imgtype = detect_picture(data)
    # print(imgtype)
    if imgtype == "jpeg" or imgtype == "png":
        with open(imagePath, 'wb') as f:
            f.write(data)
            return 

    raise ValueError("error")

def detect_picture(data):
    """
    检测图片的类型
    :param file: 图片二进制
    :return: 
    """
    
    if data[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'

    elif data.startswith(b'\211PNG\r\n\032\n'):
        return 'png'

    elif data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

    elif data[:2] in (b'MM', b'II'):
        return 'tiff'

    elif data.startswith(b'\001\332'):
        return 'rgb'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'14' and data[2] in b' \t\n\r':
        return 'pbm'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'25' and data[2] in b' \t\n\r':
        return 'pgm'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'36' and data[2] in b' \t\n\r':
        return 'ppm'

    elif data.startswith(b'\x59\xA6\x6A\x95'):
        return 'rast'

    elif data.startswith(b'#define '):
        return 'xbm'

    elif data.startswith(b'BM'):
        return 'bmp'

    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'webp'

    elif data.startswith(b'\x76\x2f\x31\x01'):
        return 'exr'

    else:
        return None


def download_from_file(savePath, csvFile, download_num = 100000):
    reader = csv.reader(open(csvFile, 'r', encoding="utf-8"))
    captions_map = []
    for idx, line in tqdm(enumerate(reader), total=download_num):
        if idx == 0:
            continue
        if idx > download_num:
            break
        url = line[0]
        caption = line[1]
        saveName = "{:08d}.jpg".format(idx)

        try:
            get_response(savePath + saveName, url)
        except:
            continue
        captions_map.append([caption, saveName])

    return captions_map

def main():
    # get_response("../data/image.jpg", imgPath)
    ## 下载网络图片
    captions_map = download_from_file("F:/dataset/wukong_release/image/", "F:/dataset/wukong_release/wukong_release/wukong_100m_0.csv")
    with open("../data/image_caption.txt", 'w', encoding="utf-8") as f:
        for CM in captions_map:
            f.write(str(CM) + '\n')

if __name__ == "__main__":
    main()
