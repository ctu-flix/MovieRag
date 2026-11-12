"""MovieRAG Phase A Extraction Pipeline Package.

Provides modular, checkpointed CLI and notebook orchestration scripts for:
- Stage A1 (stage1_scenes.py): Shot boundary detection & 15-30s scene windowing.
- Stage A2 (stage2_charbank.py): Semi-automated visual character bank & DINOv2 contrastive fine-tuning.
- Stage A3 (stage3_visual_recog.py): Per-shot TGRP tracking, DINOv2 track ID, & adaptive annotated keyframes.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANIMATED_AD_ROOT = PROJECT_ROOT / "Animated_AD"


def ensure_animated_ad_paths() -> None:
    """Add Animated_AD module directories to sys.path if not already present."""
    candidate_dirs = [
        PROJECT_ROOT,
        ANIMATED_AD_ROOT,
        ANIMATED_AD_ROOT / "build_character_bank" / "feature_extraction",
        ANIMATED_AD_ROOT / "build_character_bank" / "build_appearance_bank",
        ANIMATED_AD_ROOT / "character_recognition" / "visual_recognition",
        ANIMATED_AD_ROOT / "app" / "AutoAD-Zero" / "stage1_qwenvl",
    ]
    for path in candidate_dirs:
        path_str = str(path)
        if os.path.isdir(path_str) and path_str not in sys.path:
            sys.path.insert(0, path_str)
