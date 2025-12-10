#!/usr/bin/env python3
"""
Kaggle Dataset Upload Script for MathLLM-TRM

사용법:
1. Kaggle API 설정: ~/.kaggle/kaggle.json 또는 환경변수
2. 실행: python upload_kaggle_dataset.py --username YOUR_KAGGLE_USERNAME

이 스크립트는:
1. 필요한 파일들을 kaggle_dataset/ 폴더에 복사
2. dataset-metadata.json 생성
3. Kaggle에 업로드
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def create_dataset_folder(base_dir: Path, output_dir: Path):
    """필요한 파일들을 dataset 폴더로 복사"""

    # 기존 폴더 삭제
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 복사할 파일/폴더 목록
    items_to_copy = [
        # 소스 코드
        ("src", "src"),
        ("eval", "eval"),

        # 학습 스크립트
        ("train_trm.py", "train_trm.py"),
        ("train_finetune.py", "train_finetune.py"),
        ("train_lora.py", "train_lora.py"),
        ("main.py", "main.py"),

        # 설정 파일
        ("pyproject.toml", "pyproject.toml"),
        ("requirements.txt", "requirements.txt"),

        # 문서
        ("CLAUDE.md", "CLAUDE.md"),
        ("TODO.md", "TODO.md"),
    ]

    copied = []
    for src_name, dst_name in items_to_copy:
        src_path = base_dir / src_name
        dst_path = output_dir / dst_name

        if not src_path.exists():
            print(f"  [SKIP] {src_name} (not found)")
            continue

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        copied.append(src_name)
        print(f"  [COPY] {src_name}")

    # __pycache__ 제거
    for pycache in output_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)

    return copied


def create_metadata(output_dir: Path, username: str, dataset_name: str, title: str):
    """dataset-metadata.json 생성"""

    metadata = {
        "title": title,
        "id": f"{username}/{dataset_name}",
        "licenses": [{"name": "Apache 2.0"}],
        "keywords": ["math", "llm", "trm", "qwen", "aimo"],
        "resources": [
            {
                "path": ".",
                "description": "MathLLM-TRM source code for training recursive transformer on math problems"
            }
        ]
    }

    metadata_path = output_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [CREATE] dataset-metadata.json")
    return metadata_path


def upload_to_kaggle(output_dir: Path, new: bool = True):
    """Kaggle에 업로드"""

    cmd = ["kaggle", "datasets", "create" if new else "version", "-p", str(output_dir)]
    if not new:
        cmd.extend(["-m", "Update source code"])

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  [SUCCESS] Upload complete!")
        print(result.stdout)
    else:
        print(f"  [ERROR] Upload failed")
        print(result.stderr)

        # 이미 존재하면 version update 시도
        if "already exists" in result.stderr and new:
            print("\n  Dataset already exists, trying to update...")
            return upload_to_kaggle(output_dir, new=False)

    return result.returncode == 0


def create_zip(output_dir: Path, zip_path: Path):
    """ZIP 파일 생성 (수동 업로드용)"""
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', output_dir)
    print(f"  [CREATE] {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Upload MathLLM code to Kaggle Dataset")
    parser.add_argument("--username", type=str, required=True,
                        help="Your Kaggle username")
    parser.add_argument("--dataset-name", type=str, default="mathllm-trm-code",
                        help="Dataset name (default: mathllm-trm-code)")
    parser.add_argument("--title", type=str, default="MathLLM-TRM Code",
                        help="Dataset title")
    parser.add_argument("--no-upload", action="store_true",
                        help="Don't upload, just create the folder and zip")
    parser.add_argument("--output-dir", type=str, default="./kaggle_dataset",
                        help="Output directory for dataset files")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 50)
    print("MathLLM Kaggle Dataset Uploader")
    print("=" * 50)

    # 1. 파일 복사
    print(f"\n[1/4] Copying files to {output_dir}...")
    create_dataset_folder(base_dir, output_dir)

    # 2. metadata 생성
    print(f"\n[2/4] Creating metadata...")
    create_metadata(output_dir, args.username, args.dataset_name, args.title)

    # 3. ZIP 생성 (백업용)
    print(f"\n[3/4] Creating ZIP archive...")
    zip_path = base_dir / f"{args.dataset_name}.zip"
    create_zip(output_dir, zip_path)

    # 4. 업로드
    if args.no_upload:
        print(f"\n[4/4] Skipping upload (--no-upload flag)")
        print(f"\n  Manual upload:")
        print(f"    1. Go to https://www.kaggle.com/datasets")
        print(f"    2. Click 'New Dataset'")
        print(f"    3. Upload {zip_path}")
    else:
        print(f"\n[4/4] Uploading to Kaggle...")
        success = upload_to_kaggle(output_dir)

        if not success:
            print(f"\n  If upload failed, you can manually upload:")
            print(f"    ZIP file: {zip_path}")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

    print(f"\nKaggle Notebook에서 사용하기:")
    print(f"```python")
    print(f"import sys")
    print(f"sys.path.insert(0, '/kaggle/input/{args.dataset_name}')")
    print(f"")
    print(f"from src.model import QwenTRM")
    print(f"from src.config import TRMConfig")
    print(f"```")


if __name__ == "__main__":
    main()
