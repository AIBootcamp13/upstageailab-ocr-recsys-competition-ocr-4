#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 후 predict 결과를 val 폴더의 이미지에 시각화하여 별도 폴더에 저장하는 스크립트

사용법:
python3 visualize_val_predictions.py --exp_name {your_exp_name}

예시:
python3 visualize_val_predictions.py --exp_name convnext_base_1020_120000
"""

import json
import os
import argparse
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

def load_prediction_results(exp_name):
    """
    예측 결과를 로드 (outputs/{exp_name}/predictions 폴더에서 JSON 찾기 - VAL 예측만)
    """
    prediction_dir = Path('outputs') / exp_name / 'predictions'

    if not prediction_dir.exists():
        print(f"예측 폴더를 찾을 수 없습니다: {prediction_dir}")
        return None

    # JSON 파일 찾기 (val_predictions.json 같은 파일명 찾기)
    json_files = list(prediction_dir.glob('*val*.json')) or list(prediction_dir.glob('*predictions*.json'))

    if not json_files:
        # 일반 JSON 파일들 중에서 찾기
        json_files = list(prediction_dir.glob('*.json'))

    if not json_files:
        print(f"예측 JSON 파일을 찾을 수 없습니다 in {prediction_dir}")
        return None

    # 가장 최신 파일 선택
    prediction_file = max(json_files, key=lambda p: p.stat().st_mtime)

    print(f"예측 파일 로드: {prediction_file}")
    with open(prediction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # JSON 포맷 변환: {"images": {filename: {words: {...}}}} -> {filename: [{"text_regions": [...]}]}
    predictions = {}
    if 'images' in data:
        for image_filename, image_data in data['images'].items():
            predictions[image_filename] = []
            if 'words' in image_data:
                for word_id, word_data in image_data['words'].items():
                    if 'points' in word_data:
                        predictions[image_filename].append({
                            'text_regions': [{'points': word_data['points']}]
                        })
    else:
        # 대안: old format 유지
        predictions = data

    return predictions

def visualize_predictions(image_path, predictions, output_path):
    """
    예측 결과를 이미지에 시각화하여 저장
    """
    # 이미지 읽기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # 폴리곤 형태의 예측 결과를 시각화
    if not predictions:
        print(f"예측 결과가 없습니다: {image_path}")
        return

    for prediction in predictions:
        if 'text_regions' in prediction and prediction['text_regions']:
            for region in prediction['text_regions']:
                points = np.array(region['points'], dtype=np.int32).reshape((-1, 1, 2))

                # 진한 하늘색, 2px 선 굵기
                color = (0, 180, 0)  # BGR 형식: 차분한 녹색 (0, 180, 0)
                thickness = 2

                # 폴리곤 그리기
                cv2.polylines(image, [points], True, color, thickness)

    # 출력 폴더 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미지 저장
    success = cv2.imwrite(str(output_path), image)
    if success:
        print(f"시각화 저장: {output_path}")
    else:
        print(f"저장 실패: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="실험 후 예측 결과를 val 이미지에 시각화")
    parser.add_argument('--exp_name', type=str, required=True,
                       help='실험 이름 (예: convnext_base_1020_120000)')
    args = parser.parse_args()

    exp_name = args.exp_name

    # 경로 설정
    base_path = Path('data/datasets')
    images_dir = base_path / 'images' / 'val'  # VAL 이미지 경로

    print(f"이미지 디렉토리: {images_dir}")
    print(f"디렉토리 존재: {images_dir.exists()}")
    if images_dir.exists():
        print(f"이미지 파일 개수: {len(list(images_dir.glob('*.jpg')))}")
    output_dir = Path('visualized_predictions') / exp_name

    # 예측 결과 로드
    predictions_by_image = load_prediction_results(exp_name)
    if predictions_by_image is None:
        print("예측 결과를 로드할 수 없습니다.")
        return

    print(f"총 {len(predictions_by_image)}개 이미지의 예측 결과를 시각화합니다.")

    # 각 이미지에 대해 처리
    processed_count = 0
    for image_filename, predictions in predictions_by_image.items():
        image_path = images_dir / image_filename
        if not image_path.exists():
            print(f"원본 이미지를 찾을 수 없습니다: {image_path}")
            continue

        output_path = output_dir / image_filename
        visualize_predictions(image_path, predictions, output_path)
        processed_count += 1

    print(f"\n완료! {processed_count}개 이미지 시각화됨")
    print(f"결과 폴더: {output_dir}")

if __name__ == "__main__":
    main()
