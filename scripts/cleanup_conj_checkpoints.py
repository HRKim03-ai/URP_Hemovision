#!/usr/bin/env python3
"""
CSV 파일에서 top 5 모델을 확인하고, 나머지 체크포인트를 삭제합니다.
"""

import csv
import os
from pathlib import Path


def get_top5_backbones(csv_path: str) -> list[str]:
    """CSV에서 MAE 기준 top 5 backbone 이름을 반환합니다."""
    backbones = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # MAE 기준으로 정렬 (낮을수록 좋음)
        sorted_rows = sorted(rows, key=lambda x: float(x['mae']))
        
        # Top 5 추출
        for row in sorted_rows[:5]:
            backbone_name = row['backbone_name']
            mae = row['mae']
            r2 = row['r2']
            print(f"  Top 5: {backbone_name} (MAE: {mae}, R2: {r2})")
            backbones.append(backbone_name)
    
    return backbones


def cleanup_checkpoints(checkpoint_dir: str, keep_backbones: list[str], dry_run: bool = False):
    """지정된 backbone들의 체크포인트만 남기고 나머지를 삭제합니다."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # keep_backbones를 파일명 형식으로 변환 (슬래시를 언더스코어로)
    keep_backbone_patterns = [bb.replace('/', '_') for bb in keep_backbones]
    
    deleted_count = 0
    space_freed_mb = 0
    kept_count = 0
    
    # conj 디렉토리 내의 모든 체크포인트 찾기
    for ckpt_file in checkpoint_dir.rglob("*.pt"):
        # 파일명에서 backbone 이름 추출
        # 형식: conj_{backbone}_epoch{epoch}.pt
        filename = ckpt_file.name
        if not filename.startswith("conj_"):
            continue
        
        # backbone 이름 추출
        parts = filename.replace("conj_", "").split("_epoch")
        if len(parts) < 2:
            continue
        
        backbone_in_file = "_".join(parts[0].split("_")[:-1] if len(parts[0].split("_")) > 1 else [parts[0]])
        # timm_ 접두사가 있으면 제거하고 비교
        if backbone_in_file.startswith("timm_"):
            backbone_in_file = backbone_in_file[5:]  # "timm_" 제거
        
        # keep_backbones와 매칭 확인
        should_keep = False
        for keep_pattern in keep_backbone_patterns:
            # timm_ 접두사 제거 후 비교
            keep_pattern_clean = keep_pattern.replace("timm_", "")
            if keep_pattern_clean in filename or backbone_in_file in keep_pattern_clean:
                should_keep = True
                break
        
        if should_keep:
            kept_count += 1
            print(f"  KEEP: {filename}")
        else:
            size_mb = ckpt_file.stat().st_size / (1024 * 1024)
            if not dry_run:
                ckpt_file.unlink()
            print(f"  {'[DRY RUN] ' if dry_run else ''}DELETE: {filename} ({size_mb:.1f} MB)")
            deleted_count += 1
            space_freed_mb += size_mb
    
    return deleted_count, space_freed_mb, kept_count


def main():
    csv_path = "checkpoints/conj/conj_test_results_summary.csv"
    checkpoint_dir = "checkpoints/conj"
    
    print("=" * 60)
    print("Conj 체크포인트 정리")
    print("=" * 60)
    print(f"\nCSV 파일 읽기: {csv_path}")
    
    top5_backbones = get_top5_backbones(csv_path)
    
    print(f"\nTop 5 Backbones:")
    for i, bb in enumerate(top5_backbones, 1):
        print(f"  {i}. {bb}")
    
    print(f"\n체크포인트 정리 시작...")
    print(f"디렉토리: {checkpoint_dir}")
    
    deleted, space_freed, kept = cleanup_checkpoints(
        checkpoint_dir, 
        top5_backbones, 
        dry_run=False  # 실제로 삭제
    )
    
    print(f"\n{'=' * 60}")
    print(f"결과:")
    print(f"  보관된 체크포인트: {kept}")
    print(f"  삭제된 체크포인트: {deleted}")
    print(f"  확보된 공간: {space_freed:.1f} MB ({space_freed/1024:.2f} GB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

