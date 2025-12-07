import argparse
import copy
import multiprocessing as mp
import sys
from typing import Any, Dict, List

import yaml

from eval.ensemble_predict import ensemble_predict
from eval.evaluate_fusion import evaluate_fusion
from eval.evaluate_single import evaluate_single
from train.train_fusion_phase1 import train_fusion_phase1
from train.train_fusion_phase2 import train_fusion_phase2
from train.train_single import train_single


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _split_backbones_for_gpus(
    backbone_names: List[str], num_gpus: int
) -> List[List[str]]:
    """균등하게 backbone 리스트를 GPU 수만큼 분할."""
    n = len(backbone_names)
    num_gpus = max(1, min(num_gpus, n))
    base = n // num_gpus
    rem = n % num_gpus
    splits: List[List[str]] = []
    start = 0
    for i in range(num_gpus):
        extra = 1 if i < rem else 0
        end = start + base + extra
        splits.append(backbone_names[start:end])
        start = end
    return splits


def _train_single_worker(config: Dict[str, Any], modality: str) -> None:
    """멀티프로세싱에서 호출되는 worker."""
    train_single(config, modality)


def _train_fusion_phase1_worker(config: Dict[str, Any]) -> None:
    """멀티프로세싱에서 호출되는 worker for fusion phase1."""
    train_fusion_phase1(config, device_id=config.get("device_id"))


def _train_fusion_phase2_worker(config: Dict[str, Any]) -> None:
    """멀티프로세싱에서 호출되는 worker for fusion phase2."""
    train_fusion_phase2(config, device_id=config.get("device_id"))


def run_train_single_multi_gpu(
    config: Dict[str, Any], modality: str, num_gpus: int
) -> None:
    """백본 리스트를 여러 GPU에 자동 분배해서 병렬 학습."""
    backbone_names = config.get("backbone_names")
    if backbone_names is None:
        backbone_names = [config["backbone_name"]]

    splits = _split_backbones_for_gpus(backbone_names, num_gpus)

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []

    for gpu_id, names in enumerate(splits):
        if not names:
            continue
        cfg = copy.deepcopy(config)
        cfg["backbone_names"] = names
        cfg["device_id"] = gpu_id  # train_single 내부에서 사용
        p = ctx.Process(target=_train_single_worker, args=(cfg, modality))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def run_train_fusion_phase1_multi_gpu(
    config: Dict[str, Any], num_gpus: int
) -> None:
    """Fold와 backbone pair 조합을 여러 GPU에 자동 분배해서 병렬 학습."""
    from datasets.fusion_dataset import load_fusion_metadata
    from utils.cv_split import create_fusion_splits
    import pandas as pd
    
    # Load samples to get splits
    samples = load_fusion_metadata(config["fusion_metadata_csv"])
    df = pd.DataFrame({"patient_id": [s.patient_id for s in samples]})
    splits = create_fusion_splits(df, patient_col="patient_id")
    
    nail_backbone_names = config["nail_backbones"]
    conj_backbone_names = config["conj_backbones"]
    
    # Generate all tasks: (fold_idx, nail_name, conj_name)
    all_tasks = []
    for fold_idx in range(len(splits.folds)):
        for nail_name in nail_backbone_names:
            for conj_name in conj_backbone_names:
                all_tasks.append((fold_idx, nail_name, conj_name))
    
    # Split tasks across GPUs
    n = len(all_tasks)
    num_gpus = max(1, min(num_gpus, n))
    base = n // num_gpus
    rem = n % num_gpus
    task_splits: List[List[tuple]] = []
    start = 0
    for i in range(num_gpus):
        extra = 1 if i < rem else 0
        end = start + base + extra
        task_splits.append(all_tasks[start:end])
        start = end
    
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    
    for gpu_id, tasks in enumerate(task_splits):
        if not tasks:
            continue
        cfg = copy.deepcopy(config)
        cfg["assigned_tasks"] = tasks
        cfg["device_id"] = gpu_id
        p = ctx.Process(target=_train_fusion_phase1_worker, args=(cfg,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


def run_train_fusion_phase2_multi_gpu(
    config: Dict[str, Any], num_gpus: int
) -> None:
    """Fold와 backbone pair 조합을 여러 GPU에 자동 분배해서 병렬 학습 (Phase 2)."""
    from datasets.fusion_dataset import load_fusion_metadata
    from utils.cv_split import create_fusion_splits
    import pandas as pd
    
    # Load samples to get splits
    samples = load_fusion_metadata(config["fusion_metadata_csv"])
    df = pd.DataFrame({"patient_id": [s.patient_id for s in samples]})
    splits = create_fusion_splits(df, patient_col="patient_id")
    
    nail_backbone_names = config["nail_backbones"]
    conj_backbone_names = config["conj_backbones"]
    
    # Generate all tasks: (fold_idx, nail_name, conj_name)
    all_tasks = []
    for fold_idx in range(len(splits.folds)):
        for nail_name in nail_backbone_names:
            for conj_name in conj_backbone_names:
                all_tasks.append((fold_idx, nail_name, conj_name))
    
    # Split tasks across GPUs
    n = len(all_tasks)
    num_gpus = max(1, min(num_gpus, n))
    base = n // num_gpus
    rem = n % num_gpus
    task_splits: List[List[tuple]] = []
    start = 0
    for i in range(num_gpus):
        extra = 1 if i < rem else 0
        end = start + base + extra
        task_splits.append(all_tasks[start:end])
        start = end
    
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    
    for gpu_id, tasks in enumerate(task_splits):
        if not tasks:
            continue
        cfg = copy.deepcopy(config)
        cfg["assigned_tasks"] = tasks
        cfg["device_id"] = gpu_id
        p = ctx.Process(target=_train_fusion_phase2_worker, args=(cfg,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="Multimodal Hb regression pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train_single",
            "train_fusion_phase1",
            "train_fusion_phase2",
            "eval_single",
            "eval_fusion",
            "ensemble",
        ],
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--modality",
        type=str,
        choices=["nail", "conj"],
        help="Required for single-modality modes",
    )
    # 병렬 실행 시 backbone 리스트를 분할해서 사용할 때 사용 (기존 CLI 방식)
    parser.add_argument(
        "--backbone_start",
        type=int,
        default=None,
        help="Optional start index (inclusive) into backbone_names list for train_single.",
    )
    parser.add_argument(
        "--backbone_end",
        type=int,
        default=None,
        help="Optional end index (exclusive) into backbone_names list for train_single.",
    )
    # 새로운 코드 레벨 멀티 GPU 분할 옵션
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="If >1, will split tasks across multiple GPUs in code. Works for train_single, train_fusion_phase1, and train_fusion_phase2.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train_single":
        if args.modality is None:
            print("--modality is required for train_single", file=sys.stderr)
            sys.exit(1)

        if args.num_gpus and args.num_gpus > 1:
            # 코드 레벨에서 백본을 GPU에 자동 분배
            run_train_single_multi_gpu(config, args.modality, args.num_gpus)
        else:
            # 기존 단일 프로세스 / 수동 분할 방식
            train_single(
                config,
                args.modality,
                backbone_start=args.backbone_start,
                backbone_end=args.backbone_end,
            )
    elif args.mode == "train_fusion_phase1":
        # Check if we should run both versions (with and without demographics)
        run_both_demo_versions = config.get("run_both_demo_versions", False)
        # wo_demo 먼저 실행, w_demo 나중에 실행
        use_demographics_list = [False, True] if run_both_demo_versions else [config.get("use_demographics", True)]
        
        for use_demo in use_demographics_list:
            cfg = copy.deepcopy(config)
            cfg["use_demographics"] = use_demo
            demo_label = "w_demo" if use_demo else "wo_demo"
            print(f"\n{'='*60}")
            print(f"Running Phase 1 Fusion: {demo_label}")
            print(f"{'='*60}\n")
            
            if args.num_gpus and args.num_gpus > 1:
                # 코드 레벨에서 fold와 pair 조합을 GPU에 자동 분배
                run_train_fusion_phase1_multi_gpu(cfg, args.num_gpus)
            else:
                # 기존 단일 프로세스 방식
                train_fusion_phase1(cfg)
        
        # Phase 1 완료 후 Phase 2 자동 시작
        auto_start_phase2 = config.get("auto_start_phase2", True)
        if auto_start_phase2:
            phase2_config_path = config.get("phase2_config_path", "config/fusion_phase2.yaml")
            print(f"\n{'='*60}")
            print(f"Phase 1 완료! Phase 2 자동 시작...")
            print(f"{'='*60}\n")
            
            # Phase 2 config 로드
            phase2_config = load_config(phase2_config_path)
            phase2_num_gpus = config.get("phase2_num_gpus", args.num_gpus)
            
            # Phase 2 실행
            run_both_demo_versions_p2 = phase2_config.get("run_both_demo_versions", False)
            use_demographics_list_p2 = [False, True] if run_both_demo_versions_p2 else [phase2_config.get("use_demographics", True)]
            
            for use_demo in use_demographics_list_p2:
                cfg_p2 = copy.deepcopy(phase2_config)
                cfg_p2["use_demographics"] = use_demo
                demo_label = "w_demo" if use_demo else "wo_demo"
                print(f"\n{'='*60}")
                print(f"Running Phase 2 Fusion: {demo_label}")
                print(f"{'='*60}\n")
                
                if phase2_num_gpus and phase2_num_gpus > 1:
                    run_train_fusion_phase2_multi_gpu(cfg_p2, phase2_num_gpus)
                else:
                    train_fusion_phase2(cfg_p2)
    elif args.mode == "train_fusion_phase2":
        # Check if we should run both versions (with and without demographics)
        run_both_demo_versions = config.get("run_both_demo_versions", False)
        # wo_demo 먼저 실행, w_demo 나중에 실행
        use_demographics_list = [False, True] if run_both_demo_versions else [config.get("use_demographics", True)]
        
        for use_demo in use_demographics_list:
            cfg = copy.deepcopy(config)
            cfg["use_demographics"] = use_demo
            demo_label = "w_demo" if use_demo else "wo_demo"
            print(f"\n{'='*60}")
            print(f"Running Phase 2 Fusion: {demo_label}")
            print(f"{'='*60}\n")
            
            if args.num_gpus and args.num_gpus > 1:
                # 코드 레벨에서 fold와 pair 조합을 GPU에 자동 분배
                run_train_fusion_phase2_multi_gpu(cfg, args.num_gpus)
            else:
                # 기존 단일 프로세스 방식
                train_fusion_phase2(cfg)
    elif args.mode == "eval_single":
        if args.modality is None:
            print("--modality is required for eval_single", file=sys.stderr)
            sys.exit(1)
        evaluate_single(config, args.modality)
    elif args.mode == "eval_fusion":
        evaluate_fusion(config)
    elif args.mode == "ensemble":
        ensemble_predict(config)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()


