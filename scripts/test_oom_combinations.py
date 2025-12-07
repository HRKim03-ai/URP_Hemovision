#!/usr/bin/env python3
"""
OOM 테스트 스크립트: 모든 모델 조합을 배치 사이즈별로 테스트하여 OOM 발생 여부 확인
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import List, Tuple

# 프로젝트 루트를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_models import Phase1FusionModel
from datasets.fusion_dataset import load_fusion_metadata, FusionDataset
from utils.cv_split import create_fusion_splits
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms


def test_model_combination(
    nail_backbone: str,
    conj_backbone: str,
    batch_size: int,
    device: torch.device,
    demo_dim: int = 2,
) -> Tuple[bool, str]:
    """
    특정 모델 조합을 테스트하여 OOM 발생 여부 확인
    
    Returns:
        (success, error_message): 성공하면 (True, ""), 실패하면 (False, error_msg)
    """
    try:
        # 모델 생성
        model = Phase1FusionModel(
            nail_backbone_name=nail_backbone,
            conj_backbone_name=conj_backbone,
            demo_dim=demo_dim,
            pretrained=False,  # 테스트용이므로 pretrained 불필요
        ).to(device)
        
        # 더미 데이터 생성 (배치 사이즈만큼)
        nail_img = torch.randn(batch_size, 3, 224, 224).to(device)
        conj_img = torch.randn(batch_size, 3, 224, 224).to(device)
        demo = torch.randn(batch_size, demo_dim).to(device) if demo_dim > 0 else None
        
        # Forward pass 테스트
        with torch.no_grad():
            output = model(nail_img, conj_img, demo)
        
        # 메모리 정리
        del model, nail_img, conj_img, demo, output
        torch.cuda.empty_cache()
        
        return (True, "")
    
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return (False, str(e))
    except Exception as e:
        torch.cuda.empty_cache()
        return (False, f"Unexpected error: {str(e)}")


def test_all_combinations(
    nail_backbones: List[str],
    conj_backbones: List[str],
    batch_sizes: List[int],
    num_gpus: int = 4,
    demo_dim: int = 2,
) -> None:
    """
    모든 조합을 테스트하고 결과를 출력
    """
    results = []
    total_combinations = len(nail_backbones) * len(conj_backbones) * len(batch_sizes)
    current = 0
    
    print(f"=== OOM 테스트 시작 ===")
    print(f"총 조합 수: {total_combinations}")
    print(f"GPU 수: {num_gpus}")
    print(f"배치 사이즈: {batch_sizes}")
    print()
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"배치 사이즈: {batch_size}")
        print(f"{'='*60}")
        
        for nail_idx, nail_backbone in enumerate(nail_backbones):
            for conj_idx, conj_backbone in enumerate(conj_backbones):
                current += 1
                # GPU 할당 (round-robin)
                device_id = (current - 1) % num_gpus
                device = torch.device(f"cuda:{device_id}")
                
                pair_name = f"{nail_backbone} + {conj_backbone}"
                print(f"[{current}/{total_combinations}] {pair_name} (GPU {device_id}, batch={batch_size})... ", end="", flush=True)
                
                success, error_msg = test_model_combination(
                    nail_backbone=nail_backbone,
                    conj_backbone=conj_backbone,
                    batch_size=batch_size,
                    device=device,
                    demo_dim=demo_dim,
                )
                
                if success:
                    print("✓ OK")
                    results.append({
                        "nail_backbone": nail_backbone,
                        "conj_backbone": conj_backbone,
                        "batch_size": batch_size,
                        "status": "OK",
                        "error": "",
                    })
                else:
                    print(f"✗ OOM")
                    results.append({
                        "nail_backbone": nail_backbone,
                        "conj_backbone": conj_backbone,
                        "batch_size": batch_size,
                        "status": "OOM",
                        "error": error_msg[:100],  # 처음 100자만
                    })
    
    # 결과 요약
    print(f"\n\n{'='*60}")
    print("테스트 결과 요약")
    print(f"{'='*60}\n")
    
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r["batch_size"] == batch_size]
        oom_count = sum(1 for r in batch_results if r["status"] == "OOM")
        ok_count = len(batch_results) - oom_count
        
        print(f"배치 사이즈 {batch_size}:")
        print(f"  ✓ 성공: {ok_count}/{len(batch_results)}")
        print(f"  ✗ OOM: {oom_count}/{len(batch_results)}")
        
        if oom_count > 0:
            print(f"  OOM 발생 조합:")
            for r in batch_results:
                if r["status"] == "OOM":
                    print(f"    - {r['nail_backbone']} + {r['conj_backbone']}")
        print()
    
    # 결과를 파일로 저장
    results_file = PROJECT_ROOT / "logs" / "oom_test_results.txt"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        f.write("OOM 테스트 결과\n")
        f.write("=" * 60 + "\n\n")
        
        for batch_size in batch_sizes:
            f.write(f"배치 사이즈 {batch_size}:\n")
            f.write("-" * 60 + "\n")
            batch_results = [r for r in results if r["batch_size"] == batch_size]
            for r in batch_results:
                status_symbol = "✓" if r["status"] == "OK" else "✗"
                f.write(f"{status_symbol} {r['nail_backbone']} + {r['conj_backbone']}\n")
                if r["status"] == "OOM":
                    f.write(f"    Error: {r['error']}\n")
            f.write("\n")
    
    print(f"상세 결과가 저장되었습니다: {results_file}")


def main():
    # 설정 파일 로드
    config_path = PROJECT_ROOT / "config" / "fusion_phase1.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    nail_backbones = config["nail_backbones"]
    conj_backbones = config["conj_backbones"]
    
    # 테스트할 배치 사이즈들
    batch_sizes = [64, 128, 256]
    
    # GPU 수
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU: {num_gpus}개")
    
    # demo_dim (demographic features 사용 여부)
    use_demographics = config.get("use_demographics", True)
    demo_dim = 2 if use_demographics else 0
    
    test_all_combinations(
        nail_backbones=nail_backbones,
        conj_backbones=conj_backbones,
        batch_sizes=batch_sizes,
        num_gpus=num_gpus,
        demo_dim=demo_dim,
    )


if __name__ == "__main__":
    main()

