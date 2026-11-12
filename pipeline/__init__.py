"""MovieRAG Phase A Extraction Pipeline Package.

Provides modular, checkpointed CLI and notebook orchestration scripts for:
- Stage A1 (stage1_scenes.py): Shot boundary detection & 15-30s scene windowing.
- Stage A2 (stage2_charbank.py): Semi-automated visual character bank & DINOv2 contrastive fine-tuning.
- Stage A3 (stage3_visual_recog.py): Per-shot TGRP tracking, DINOv2 track ID, & adaptive annotated keyframes.
"""

import os
import site
import ssl
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANIMATED_AD_ROOT = PROJECT_ROOT / "Animated_AD"


def configure_ssl_and_Editable_sam2() -> None:
    """Bypass expired SSL certs on dl.fbaipublicfiles.com and ensure /content/sam2 is importable."""
    try:
        unverified_ctx = ssl._create_unverified_context()
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.HTTPSHandler(context=unverified_ctx))
        )
    except Exception:
        pass

    sam2_repo = "/content/sam2"
    if os.path.isdir(sam2_repo):
        # Clear any stale broken namespace module for 'sam2'
        sam2_mod = sys.modules.get("sam2")
        if sam2_mod is not None and not hasattr(sam2_mod, "__file__"):
            for mod_name in [k for k in list(sys.modules) if k == "sam2" or k.startswith("sam2.")]:
                sys.modules.pop(mod_name, None)
        try:
            site.main()
        except Exception:
            pass
        if sam2_repo not in sys.path:
            sys.path.insert(0, sam2_repo)


def patch_owlv2_processor() -> None:
    """Ensure Owlv2Processor has post_process_object_detection on transformers >= 4.48 / 5.x."""
    try:
        from transformers import Owlv2Processor

        if not hasattr(Owlv2Processor, "post_process_object_detection"):

            def _post_process_object_detection(self, outputs, threshold=0.1, target_sizes=None):
                if hasattr(self, "post_process_grounded_object_detection"):
                    return self.post_process_grounded_object_detection(
                        outputs=outputs, threshold=threshold, target_sizes=target_sizes
                    )
                return self.image_processor.post_process_object_detection(
                    outputs=outputs, threshold=threshold, target_sizes=target_sizes
                )

            Owlv2Processor.post_process_object_detection = _post_process_object_detection
    except Exception:
        pass


def ensure_animated_ad_paths() -> None:
    """Add Animated_AD module directories to sys.path in priority order."""
    configure_ssl_and_Editable_sam2()

    # Ordered from lowest to highest priority since sys.path.insert(0, ...) prepends
    candidate_dirs = [
        ANIMATED_AD_ROOT / "build_character_bank" / "feature_extraction",
        ANIMATED_AD_ROOT / "character_recognition" / "visual_recognition",
        ANIMATED_AD_ROOT,
        PROJECT_ROOT,
    ]
    for path in candidate_dirs:
        path_str = str(path)
        if os.path.isdir(path_str) and path_str not in sys.path:
            sys.path.insert(0, path_str)
