#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ExifTags

def get_exif_orientation(image_path):
    """
    이미지 파일의 EXIF orientation 값을 가져옴
    """
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                    return value
    except Exception as e:
        print(f"EXIF 읽기 에러 {image_path}: {e}")
    return 1  # 기본값: 회전 없음

def correct_orientation(image, orientation):
    """
    이미지의 EXIF orientation 정보를 바탕으로 이미지 회전 (정방향으로 바로 세우기)
    """
    if orientation == 2:
        image = cv2.flip(image, 1)  # 좌우 반전
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)  # 180도 회전
    elif orientation == 4:
        image = cv2.flip(image, 0)  # 상하 반전
    elif orientation == 5:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 1)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 90도 회전 (시계방향)
    elif orientation == 7:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270도 회전

    return image

def visualize_bboxes(image_path, bboxes, output_path):
    """
    이미지에 bounding box들을 시각화하고 저장
    """
    # 이미지 읽기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # 폴리곤 형태의 bounding box 그리기 (회전 없이 원본 이미지에 그대로 표시)
    for word_id, word_data in bboxes.items():
        # Points 그대로 사용
        points = np.array(word_data['points'], dtype=np.int32).reshape((-1, 1, 2))

        # 폴리곤 색상 - 진한 하늘색
        color = (255, 128, 0)  # 진한 하늘색 (BGR)
        thickness = 2  # 선 굵기 조절 (1.5 두껍게)

        # 폴리곤 그리기 (True: closed polygon)
        cv2.polylines(image, [points], True, color, thickness)

        # 텍스트 표시 제거

    # 출력 폴더 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미지 저장
    success = cv2.imwrite(str(output_path), image)
    if success:
        print(f"저장됨: {output_path}")
    else:
        print(f"저장 실패: {output_path}")

def main():
    # 경로 설정
    base_path = Path('data/datasets')
    images_dir = base_path / 'images' / 'val'
    annotations_file = base_path / 'jsons' / 'val.json'
    output_dir = Path('visualized_val')

    # 어노테이션 로드
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"총 {len(annotations['images'])}개 이미지 처리")

    # 각 이미지에 대해 처리
    processed_count = 0
    for image_filename, image_data in annotations['images'].items():
        image_path = images_dir / image_filename
        if not image_path.exists():
            print(f"이미지 없음: {image_path}")
            continue

        output_path = output_dir / image_filename
        bboxes = image_data['words']

        try:
            visualize_bboxes(image_path, bboxes, output_path)
            processed_count += 1

            # 진행상황 표시 (10개마다)
            if processed_count % 10 == 0:
                print(f"진행중... {processed_count}/{len(annotations['images'])}")

        except Exception as e:
            print(f"처리 에러 {image_filename}: {e}")
            continue

    print(f"\n완료! {processed_count}개 이미지 처리됨")
    print(f"결과 폴더: {output_dir}")

if __name__ == "__main__":
    main()
