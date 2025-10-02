import os
from PIL import Image

def convert_png_to_jpg_black_bg(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                png_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'
                try:
                    img = Image.open(png_path)
                    if img.mode == 'RGBA':
                        # 검정 배경 생성
                        bg = Image.new('RGB', img.size, (0, 0, 0))
                        bg.paste(img, (0, 0), img)  # alpha channel 사용
                        img = bg
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(jpg_path, 'JPEG')
                    print(f"Converted {png_path} to {jpg_path}")
                    os.remove(png_path)
                    print(f"Removed original {png_path}")
                except Exception as e:
                    print(f"Error converting {png_path}: {e}")

if __name__ == "__main__":
    convert_png_to_jpg_black_bg('data/datasets/images')
