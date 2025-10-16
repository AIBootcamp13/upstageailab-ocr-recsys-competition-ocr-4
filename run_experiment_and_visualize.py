#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험을 실행하고 VAL 예측 결과를 저장한 후 자동으로 시각화를 수행하는 통합 스크립트

사용법:
python3 run_experiment_and_visualize.py --preset example
python3 run_experiment_and_visualize.py --preset example --exp_suffix "custom_run"

리뷰:
1. 실험 실행 (train/test)
2. VAL 예측 실행 및 결과 저장
3. VAL 예측 시각화 생성
4. 모든 결과 정리 출력
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path
from datetime import datetime
from pytz import timezone as pytz_timezone

def run_experiment(preset, exp_suffix=None):
    """실험 실행 (train + test + submit)"""
    print("🚀 실험 실행 시작...")

    # 타임스탬프 생성
    kst_tz = pytz_timezone('Asia/Seoul')
    timestamp = datetime.now(kst_tz).strftime("%m%d_%H%M%S")
    exp_name_suffix = f"_{timestamp}" if exp_suffix is None else f"_{exp_suffix}_{timestamp}"

    # 기존 run_pipeline.py 실행
    cmd = ["uv", "run", "python", "code/baseline_code/runners/run_pipeline.py", preset, exp_name_suffix]
    result = subprocess.run(cmd, check=True, env=dict(os.environ, HYDRA_FULL_ERROR='1'))

    if result.returncode != 0:
        print("❌ 실험 실행 실패")
        return None

    # 생성된 exp_name 추출 (outputs 폴더에서 가장 최근 생성된 폴더)
    outputs_dir = Path("outputs")
    subdirs = [d for d in outputs_dir.glob("*") if d.is_dir() and any((d / "checkpoints").rglob("*.ckpt"))]
    latest_experiment = max(subdirs, key=lambda d: d.stat().st_mtime)
    exp_name = latest_experiment.name

    print(f"✅ 실험 완료: {exp_name}")
    return exp_name

def run_val_prediction(exp_name):
    """VAL 예측 실행"""
    print(f"🔍 VAL 예측 실행 ({exp_name})...")

    # Best checkpoint 경로
    checkpoint_path = f"outputs/{exp_name}/checkpoints/best/model.ckpt"

    if not Path(checkpoint_path).exists():
        print("⚠️ Best checkpoint가 존재하지 않습니다. 최신 checkpoint 검색...")
        # Fallback: 가장 최근/최고 hmean checkpoint 찾기
        checkpoint_dir = Path(f"outputs/{exp_name}/checkpoints")
        ckpts = list(checkpoint_dir.rglob("*.ckpt"))
        if not ckpts:
            print("❌ 유효한 checkpoint를 찾을 수 없습니다.")
            return False

        def extract_hmean(path):
            match = re.search(r'hmean=([0-9]+\.\d+)', path.name)
            return float(match.group(1)) if match else 0.0

        best_ckpt = max(ckpts, key=extract_hmean)
        checkpoint_path = str(best_ckpt)
        print(f"📍 대안 checkpoint 사용: {best_ckpt.name}")

    # VAL 예측 실행
    cmd = ["uv", "run", "python", "code/baseline_code/runners/predict_val_only.py",
           "preset=example", f"checkpoint_path='{checkpoint_path}'"]

    try:
        result = subprocess.run(cmd, check=True, env=dict(os.environ, HYDRA_FULL_ERROR='1'))
        print("✅ VAL 예측 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ VAL 예측 실패: {e}")
        return False

def run_visualization(exp_name):
    """VAL 예측 결과 시각화"""
    print(f"🎨 VAL 예측 시각화 실행 ({exp_name})...")

    cmd = ["python3", "visualize_val_predictions.py", "--exp_name", exp_name]

    try:
        result = subprocess.run(cmd, check=True)
        print("✅ 시각화 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 시각화 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="실험 실행 + VAL 예측 + 시각화 통합 스크립트")
    parser.add_argument('--preset', type=str, required=True,
                       help='프리셋 이름 (예: example)')
    parser.add_argument('--exp_suffix', type=str, default=None,
                       help='실험 이름 접미사 (선택사항, 미지정시 타임스탬프만 사용)')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='기존 실험 이름 지정 (선택사항, 지정시 실험 재실행 없이 VAL 예측/시각화만 수행)')

    args = parser.parse_args()

    # 메인 워크플로우
    print("🔬 OCR 실험 자동화 스크립트 시작")
    print("="*50)

    # 기존 실험이 지정되었으면 재실행하지 않음
    if args.exp_name:
        exp_name = args.exp_name
        print(f"📋 기존 실험 사용: {exp_name}")
        if not Path(f"outputs/{exp_name}/checkpoints/best/model.ckpt").exists():
            print(f"❌ {exp_name} 실험의 best checkpoint가 존재하지 않습니다.")
            sys.exit(1)
    else:
        # 1. 실험 실행
        exp_name = run_experiment(args.preset, args.exp_suffix)
        if not exp_name:
            sys.exit(1)

    print()

    # 2. VAL 예측
    if not run_val_prediction(exp_name):
        sys.exit(1)

    print()

    # 3. 시각화
    if not run_visualization(exp_name):
        sys.exit(1)

    print()
    print("="*50)
    print("🎉 모든 과정 완료!")
    print()

    # 실험 모델 정보 분석 및 출력
    model_info = analyze_experiment_model(exp_name)
    print("🏷️  실험 모델 정보:")
    if model_info['preset']:
        print(f"   Preset: {model_info['preset']}")
    if model_info['encoder']:
        print(f"   Encoder: {model_info['encoder']}")
    if model_info['decoder']:
        print(f"   Decoder: {model_info['decoder']}")
    if model_info['head']:
        print(f"   Head: {model_info['head']}")
    if model_info['best_checkpoint']:
        print(f"   Checkpoint: {model_info['best_checkpoint']}")
        # best accuracy 추출
        ckpt_name = model_info['best_checkpoint']
        if 'hmean=' in ckpt_name:
            hmean = ckpt_name.split('hmean=')[1].split('-')[0]
            print(f"   Best hmean: {hmean}")
    print()

    print(f"📊 실험 이름: {exp_name}")
    print("📂 생성된 폴더들:")
    print(f"   ├── outputs/{exp_name}/")
    print(f"   |   ├── predictions/          # VAL 예측 결과 JSON")
    print(f"   |   ├── submissions/         # 대회 제출 파일들")
    print(f"   |   └── checkpoints/         # 학습된 모델들")
    print(f"   └── visualized_predictions/{exp_name}/")
    print("                               # VAL 예측 시각화 이미지들")
    print()
    print("💡 결과 파일들 확인법:")
    print(f"   - GT vs 예측 비교: visualized_val/ vs visualized_predictions/{exp_name}/")
    print("   - 예측 성능: 터미널 출력 참고")
    print()
    print("🚀 다음 실험을 위해 언제든 명령어 다시 실행하세요!")

def analyze_experiment_model(exp_name):
    """실험 이름을 기반으로 모델 정보를 분석"""
    model_info = {
        'preset': None,
        'encoder': None,
        'decoder': None,
        'head': None,
        'best_checkpoint': None
    }

    # exp_name에서 모델 정보 추출 (convnext_base_1014_152341 등의 형태)
    parts = exp_name.split('_')

    # 현재는 exp_name에서만 정보 추출하므로 hydra config에서 모델 정보를 가져와야 함
    config_file = f"outputs/{exp_name}/.hydra/config.yaml"
    if Path(config_file).exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # 모델 정보 추출
            if 'models' in config:
                if 'encoder' in config['models'] and 'backbone' in config['models']['encoder']:
                    model_info['encoder'] = config['models']['encoder']['backbone']
                if 'decoder' in config['models'] and '_target_' in config['models']['decoder']:
                    model_info['decoder'] = config['models']['decoder']['_target_'].split('.')[-1]
                if 'head' in config['models'] and '_target_' in config['models']['head']:
                    model_info['head'] = config['models']['head']['_target_'].split('.')[-1]

        except Exception as e:
            print(f"Config 분석 실패: {e}")

    # Best checkpoint 정보
    best_ckpt = Path(f"outputs/{exp_name}/checkpoints/best/model.ckpt")
    epoch_ckpts = list(Path(f"outputs/{exp_name}/checkpoints").glob("epoch=*-*.ckpt"))

    if best_ckpt.exists():
        model_info['best_checkpoint'] = "best/model.ckpt"
    elif epoch_ckpts:
        # 최고 hmean epoch 찾기
        def get_hmean(ckpt_path):
            try:
                return float(str(ckpt_path).split('hmean=')[1].split('-')[0])
            except:
                return 0.0

        best_epoch = max(epoch_ckpts, key=get_hmean)
        model_info['best_checkpoint'] = best_epoch.name

    return model_info

if __name__ == "__main__":
    main()
