#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT (Ground Truth)와 예측 결과를 동시에 한 이미지에 시각화
GT: 파랑색 폴리곤
예측: 녹색 폴리곤

사용법:
python3 visualize_combined.py --exp_name hrnet_w44.ms_in1k_1024_brcn_roate_1013_203259 --num_images 10
"""

import json
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

def load_gt_and_prediction(exp_name):
    """
    GT와 예측 결과 로드
    """
    # GT JSON 경로 찾기
    jsons_dir = Path('data/datasets/jsons')
    gt_files = list(jsons_dir.glob('*val*.json')) + list(jsons_dir.glob('*.json'))
    if gt_files:
        gt_file = gt_files[0]  # 첫 번째 찾은 것 사용
    else:
        print("GT JSON 파일을 찾을 수 없습니다.")
        return None, None

    # 예측 결과 로드
    prediction_dir = Path('outputs') / exp_name / 'predictions'
    if not prediction_dir.exists():
        print(f"예측 폴더를 찾을 수 없습니다: {prediction_dir}")
        return None, None

    prediction_files = list(prediction_dir.glob('*.json'))
    if not prediction_files:
        print(f"예측 JSON 파일을 찾을 수 없습니다 in {prediction_dir}")
        return None, None

    prediction_file = max(prediction_files, key=lambda p: p.stat().st_mtime)

    print(f"GT 파일: {gt_file}")
    print(f"예측 파일: {prediction_file}")

    # GT 로드 및 변환 (words -> 시각화 포맷)
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_raw_data = json.load(f)

    # GT 변환 (images -> filename format)
    gt_data = {}
    if 'images' in gt_raw_data:
        for image_filename, image_data in gt_raw_data['images'].items():
            gt_data[image_filename] = []
            if 'words' in image_data:
                for word_id, word_data in image_data['words'].items():
                    if 'points' in word_data:
                        gt_data[image_filename].append({
                            'text_regions': [{'points': word_data['points']}]
                        })

    # 예측 결과 로드 및 변환
    with open(prediction_file, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)

    # 예측 결과 변환 (대회 포맷 -> 시각화 포맷)
    predictions = {}
    if 'images' in prediction_data:
        for image_filename, image_data in prediction_data['images'].items():
            predictions[image_filename] = []
            if 'words' in image_data:
                for word_id, word_data in image_data['words'].items():
                    if 'points' in word_data:
                        predictions[image_filename].append({
                            'text_regions': [{'points': word_data['points']}]
                        })

    return gt_data, predictions

def visualize_combined(image_path, gt_data, prediction_data, output_path, image_filename):
    """
    GT와 예측 결과를 동시에 시각화
    GT: 파랑색, 예측: 녹색
    """
    # 이미지 읽기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # GT 폴리곤 그리기 (파랑색)
    if image_filename in gt_data and gt_data[image_filename]:
        for annotation in gt_data[image_filename]:
            if 'text_regions' in annotation and annotation['text_regions']:
                points = np.array(annotation['text_regions'][0]['points'], dtype=np.int32).reshape((-1, 1, 2))

                # 파랑색, 2px 선 굵기
                cv2.polylines(image, [points], True, (255, 0, 0), 2)  # GT: Blue (BGR: (255, 0, 0))

    # 예측 폴리곤 그리기 (녹색)
    if image_filename in prediction_data and prediction_data[image_filename]:
        for prediction in prediction_data[image_filename]:
            if 'text_regions' in prediction and prediction['text_regions']:
                for region in prediction['text_regions']:
                    points = np.array(region['points'], dtype=np.int32).reshape((-1, 1, 2))

                    # 녹색, 2px 선 굵기
                    cv2.polylines(image, [points], True, (0, 180, 0), 2)  # 예측: Green (BGR: (0, 180, 0))

    # 범례 추가 (상단 왼쪽)
    legend_x, legend_y = 10, 30
    cv2.rectangle(image, (legend_x, legend_y), (legend_x+250, legend_y+50), (255, 255, 255), -1)  # 배경
    cv2.rectangle(image, (legend_x+5, legend_y+10), (legend_x+20, legend_y+25), (255, 0, 0), -1)  # 파랑색 사각형
    cv2.rectangle(image, (legend_x+125, legend_y+10), (legend_x+140, legend_y+25), (0, 180, 0), -1)  # 녹색 사각형
    cv2.putText(image, "GT (Blue)", (legend_x+25, legend_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(image, "Prediction (Green)", (legend_x+145, legend_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)

    # 출력 폴더 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미지 저장
    success = cv2.imwrite(str(output_path), image)
    if success:
        print(f"통합 시각화 저장: {output_path}")
    else:
        print(f"저장 실패: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="GT와 예측 결과를 동시에 시각화")
    parser.add_argument('--exp_name', type=str, required=True,
                       help='실험 이름 (예: hrnet_w44.ms_in1k_1024_brcn_roate_1013_203259)')
    parser.add_argument('--num_images', type=int, default=None,
                       help='시각화할 이미지 개수 (기본: None, 전체 val 이미지 처리)')

    args = parser.parse_args()

    exp_name = args.exp_name
    num_images = args.num_images

    # 경로 설정
    images_dir = Path('data/datasets/images/val')
    output_dir = Path('visualized_combined') / exp_name

    print(f"이미지 디렉토리: {images_dir}")
    if images_dir.exists():
        print(f"이미지 파일 개수: {len(list(images_dir.glob('*.jpg')))}")

    # GT와 예측 데이터 로드
    gt_data, prediction_data = load_gt_and_prediction(exp_name)
    if gt_data is None or prediction_data is None:
        print("데이터 로드 실패")
        return

    print(f"GT 이미지 수: {len(gt_data)}")
    print(f"예측 이미지 수: {len(prediction_data)}")
    if num_images is None:
        num_images = len(gt_data)  # 전체 이미지 처리
    print(f"시각화할 이미지 수: {min(num_images, len(gt_data), len(prediction_data))}")

    # 공통 이미지 찾기 및 시각화
    processed_count = 0
    for image_filename in gt_data.keys():
        if image_filename in prediction_data and processed_count < num_images:
            image_path = images_dir / image_filename
            if not image_path.exists():
                continue

            output_path = output_dir / image_filename
            visualize_combined(image_path, gt_data, prediction_data, output_path, image_filename)
            processed_count += 1

    print(f"\n완료! {processed_count}개 이미지 통합 시각화됨")
    print(f"결과 폴더: {output_dir}")
    print("\n색상 구분:")
    print("🟦 파랑색 폴리곤: GT (Ground Truth - 정답)")
    print("🟩 녹색 폴리곤: Prediction (예측 결과)")

if __name__ == "__main__":
    main()
