#!/usr/bin/env python3
import subprocess
import sys
import re
import os
from datetime import datetime
from pathlib import Path
from pytz import timezone as pytz_timezone
from omegaconf import OmegaConf
def main():
    if len(sys.argv) > 2 and sys.argv[2] == '--resume':
        preset = sys.argv[1] if len(sys.argv) > 1 else "example"
        start_step = 2
        # config에서 base exp_name 가져오기 (csv filename용)
        config_path = Path('code/baseline_code/configs/preset', f'{preset}.yaml')
        if config_path.exists():
            try:
                config = OmegaConf.load(str(config_path))
                base_exp_name = getattr(config, 'exp_name', 'convnext_base')
                if base_exp_name is None:
                    base_exp_name = 'convnext_base'
            except:
                base_exp_name = 'convnext_base'
        else:
            base_exp_name = 'convnext_base'
        # 기존 exp_name 찾기 (checkpoint 있는 최근 실험)
        outputs_dir = Path("outputs")
        subdirs = [d for d in outputs_dir.glob("*") if d.is_dir() and any((d / "checkpoints").rglob("*.ckpt"))]
        if not subdirs:
            print("checkpoint가 있는 이전 실험 결과를 찾을 수 없습니다. outputs/ 디렉터리를 확인하거나 새로 학습을 진행하세요.")
            return
        latest_exp = max(subdirs, key=lambda d: d.stat().st_mtime)
        exp_name = latest_exp.name
        # best 중심, 없으면 최대 hmean checkpoint 선택
        ckpt_dir = latest_exp / "checkpoints"
        best_path = ckpt_dir / "best" / "model.ckpt"
        if best_path.exists():
            checkpoint_path = str(best_path)
            print(f"초기화 단계 건너뜁니다. exp_name: {exp_name} (best checkpoint)")
        else:
            ckpts = list(ckpt_dir.rglob("*.ckpt"))
            if not ckpts:
                print("no ckpts found")
                return
            def extract_hmean(path):
                match = re.search(r'-([0-9]+\.\d+)', path.name)
                if match:
                    return float(match.group(1))
                return float('-inf')
            best_ckpt = max(ckpts, key=extract_hmean)
            checkpoint_path = str(best_ckpt)
            print(f"초기화 단계 건너뜁니다. exp_name: {exp_name} (best hmean: {best_ckpt.name})")
    else:
        preset = sys.argv[1] if len(sys.argv) > 1 else "example"
        start_step = 1
        # config에서 base exp_name 가져오기
        config_path = Path('code/baseline_code/configs/preset', f'{preset}.yaml')
        if config_path.exists():
            try:
                config = OmegaConf.load(str(config_path))
                base_exp_name = getattr(config, 'exp_name', 'convnext_base')
                if base_exp_name is None:
                    base_exp_name = 'convnext_base'
            except:
                base_exp_name = 'convnext_base'
        else:
            base_exp_name = 'convnext_base'
        # 실험 중복 방지 위해 exp_name에 타임스탬프 추가 (한국시간 UTC+9)
        kst_tz = pytz_timezone('Asia/Seoul')
        timestamp = datetime.now(kst_tz).strftime("%m%d_%H%M%S")
        exp_name = f"{base_exp_name}_{timestamp}"

    # 1단계: 모델 학습
    if start_step <= 1:
        print("1단계: 모델 학습 중...")
        train_cmd = ["uv", "run", "python", "code/baseline_code/runners/train.py", f"preset={preset}", f"exp_name={exp_name}"]
        subprocess.run(train_cmd, check=True, env=dict(os.environ, HYDRA_FULL_ERROR='1'))
    else:
        print("1단계: 모델 학습 생략 (resume 모드)")

    if start_step <= 1:
        # 학습 config 기반으로 checkpoint_path 연동
        checkpoint_path = f"outputs/{exp_name}/checkpoints/best/model.ckpt"

    # 2단계: 최고 효율 checkpoint로 검증/테스트 실행
    print("2단계: 최고 효율 checkpoint로 검증/테스트 실행 중...")
    test_cmd = ["uv", "run", "python", "code/baseline_code/runners/test.py", f"preset={preset}", f"checkpoint_path='{checkpoint_path}'"]
    subprocess.run(test_cmd, check=True, env=dict(os.environ, HYDRA_FULL_ERROR='1'))

    # 3단계: 최고 효율 checkpoint로 예측 및 제출용 JSON 생성
    print("3단계: 최고 효율 checkpoint로 예측 및 제출용 JSON 생성 중...")
    predict_cmd = ["uv", "run", "python", "code/baseline_code/runners/predict.py", f"preset={preset}", f"checkpoint_path='{checkpoint_path}'", f"exp_name={exp_name}"]
    subprocess.run(predict_cmd, check=True, env=dict(os.environ, HYDRA_FULL_ERROR='1'))

    # 4단계의 경우, 예측 단계에서 JSON 파일이 생성되므로 이를 변환해야 함
    # submission_dir에서 최신 JSON 파일을 변환용으로 가정
    # submission_dir은 config에서 오는 것으로, 기본값은 outputs/${exp_name}/submissions
    submission_dir = Path(f"outputs/{exp_name}/submissions")
    if submission_dir.exists():
        json_files = list(submission_dir.glob("*.json"))
        if json_files:
            json_path = max(json_files, key=lambda p: p.stat().st_mtime)  # Most recent
            output_path = f"outputs/{exp_name}/submissions/{exp_name}_submission.csv"  # exp_name (날짜 포함)로 CSV 생성
            print("4단계: 제출 포맷 변환 중...")
            convert_cmd = ["uv", "run", "python", "code/baseline_code/ocr/utils/convert_submission.py",
                           "--json_path", str(json_path), "--output_path", output_path]
            subprocess.run(convert_cmd, check=True)
            print(f"파이프라인 완료. 최종 제출 파일: {output_path}")
        else:
            print("submissions 디렉터리에 JSON 파일이 없습니다.")
    else:
        print("제출 디렉터리를 찾을 수 없습니다. 경로를 확인하세요.")

if __name__ == "__main__":
    main()
