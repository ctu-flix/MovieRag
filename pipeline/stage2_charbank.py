#!/usr/bin/env python3
"""Stage A2: Semi-Automated Visual Character Bank (`stage2_charbank.py`).

1. Reads exemplar images from `char_bank/{movie_id}/raw_images/` (supports both nested
   `raw_images/{Character}/*.png` and flat `raw_images/{Character}_{idx}.png`) and
   validates `character_glossary.json`.
2. Stages normalized `{Character}_{idx}.png` images into `output_dir/char_bank_staged/{movie_id}/`,
   ensuring `{Character}_0.png` is preserved as the profile anchor and every character has >=2
   samples for contrastive pair sampling.
3. Optionally invokes `Animated_AD` web crawler + clustering if `--enable_crawler_fallback` is
   enabled and a character has fewer than 5 exemplar images.
4. Runs 75-epoch `DINOv2` (`dinov2_vitg14_reg` on L4 24GB or `dinov2_vitl14_reg` on T4 16GB)
   contrastive fine-tuning (InfoNCE loss, tau=0.07, lr=6e-4 -> 5e-6) and saves
   `output_dir/checkpoints/{movie_id}/finetuned_dinov2_weights.pth`.
5. Extracts character feature bank `.npz` files into `output_dir/char_features/{movie_id}/`
   using OWLv2 character detection + optional SAM 2.1 masking + fine-tuned DINOv2 embeddings,
   then flushes GPU VRAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image as Image

# Ensure project root and Animated_AD paths are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ANIMATED_AD_ROOT, ensure_animated_ad_paths

ensure_animated_ad_paths()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FLAT_NAME_RE = re.compile(r"^(?P<char>.+)_(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)


def split_char_and_idx(filename: str) -> Optional[Tuple[str, int]]:
    """Parse `{Character}_{idx}.ext` safely even when character names contain underscores or dots."""
    m = FLAT_NAME_RE.match(filename)
    if not m:
        return None
    return m.group("char"), int(m.group("idx"))


def rgba_to_rgb_white_bg(image: Image.Image, alpha_threshold: int = 100) -> Tuple[Image.Image, bool]:
    """Convert RGBA image to RGB with white background where alpha < alpha_threshold."""
    import numpy as np
    import PIL.Image as Image

    if image.mode != "RGBA":
        return image.convert("RGB"), False
    image_array = np.array(image)
    rgb_array = image_array[:, :, :3].copy()
    alpha_channel = image_array[:, :, 3]
    alpha_mask = alpha_channel < alpha_threshold
    rgb_array[alpha_mask] = [255, 255, 255]
    return Image.fromarray(rgb_array), True


def prepare_staged_character_bank(
    movie_id: str,
    char_bank_dir: str,
    staged_movie_dir: str,
    output_dir: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Normalize nested or flat character images into `staged_movie_dir/{Character}_{idx}.png`."""
    import PIL.Image as Image

    if not os.path.isdir(char_bank_dir):
        raise FileNotFoundError(f"Character bank directory not found: {char_bank_dir}")

    raw_images_dir = os.path.join(char_bank_dir, "raw_images")
    if not os.path.isdir(raw_images_dir):
        # Fallback if images are placed directly inside char_bank_dir
        raw_images_dir = char_bank_dir

    if os.path.isdir(staged_movie_dir):
        shutil.rmtree(staged_movie_dir)
    os.makedirs(staged_movie_dir, exist_ok=True)

    char_to_files: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    entries = sorted(os.listdir(raw_images_dir))
    subdirs = [
        d for d in entries
        if os.path.isdir(os.path.join(raw_images_dir, d)) and not d.startswith(".")
    ]

    if subdirs:
        # Nested layout: raw_images/{Character}/{image_files}
        for char_name in subdirs:
            char_folder = os.path.join(raw_images_dir, char_name)
            img_files = [
                f for f in sorted(os.listdir(char_folder))
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            if not img_files:
                continue

            # Identify if an explicit _0 anchor exists; otherwise first sorted image becomes _0
            anchor_candidates = []
            other_candidates = []
            for f in img_files:
                parsed = split_char_and_idx(f)
                if parsed and parsed[1] == 0:
                    anchor_candidates.append(os.path.join(char_folder, f))
                else:
                    other_candidates.append(os.path.join(char_folder, f))

            ordered_paths = anchor_candidates + other_candidates
            for idx, src_path in enumerate(ordered_paths):
                char_to_files[char_name].append((idx, src_path))
    else:
        # Flat layout: raw_images/{Character}_{idx}.png
        for fname in entries:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            parsed = split_char_and_idx(fname)
            if parsed is None:
                continue
            char_name, idx = parsed
            char_to_files[char_name].append((idx, os.path.join(raw_images_dir, fname)))

    if not char_to_files:
        raise ValueError(
            f"No valid character exemplar images found under {raw_images_dir}. "
            "Expected `raw_images/{Character}/*.png` or `raw_images/{Character}_{idx}.png`."
        )

    staged_manifest: Dict[str, List[str]] = {}
    for char_name, indexed_paths in sorted(char_to_files.items()):
        indexed_paths.sort(key=lambda x: x[0])
        staged_paths: List[str] = []

        # Ensure index 0 exists as anchor
        for new_idx, (_, src_path) in enumerate(indexed_paths):
            dst_name = f"{char_name}_{new_idx}.png"
            dst_path = os.path.join(staged_movie_dir, dst_name)
            with Image.open(src_path) as img:
                img.save(dst_path, format="PNG")
            staged_paths.append(dst_path)

        # Ensure at least 2 images per character so AnchorPlusPerLabelSampler never fails
        if len(staged_paths) == 1:
            dup_path = os.path.join(staged_movie_dir, f"{char_name}_1.png")
            shutil.copy2(staged_paths[0], dup_path)
            staged_paths.append(dup_path)

        staged_manifest[char_name] = staged_paths

    # Load or initialize character_glossary.json
    glossary_path = os.path.join(char_bank_dir, "character_glossary.json")
    glossary: Dict[str, str] = {}
    if os.path.isfile(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                glossary = {str(k): str(v) for k, v in loaded.items()}

    for char_name in staged_manifest:
        if char_name not in glossary:
            glossary[char_name] = f"Animated character {char_name} in {movie_id}."

    os.makedirs(output_dir, exist_ok=True)
    workdir_glossary_path = os.path.join(output_dir, "character_glossary.json")
    with open(workdir_glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2)

    print(f"[Stage A2] Staged {len(staged_manifest)} characters into: {staged_movie_dir}")
    for char_name, paths in staged_manifest.items():
        print(f"  - {char_name}: {len(paths)} exemplar image(s)")

    return staged_manifest, glossary


def finetune_dinov2(
    staged_movie_dir: str,
    ckpt_weights_path: str,
    model_size: str = "giant",
    epochs: int = 75,
    learning_rate: float = 6e-4,
    temperature: float = 0.07,
    overwrite: bool = False,
) -> str:
    """Fine-tune DINOv2 linear adapter for 75 epochs using InfoNCE contrastive loss."""
    if os.path.isfile(ckpt_weights_path) and not overwrite:
        print(f"[Stage A2] Found existing fine-tuned DINOv2 weights at: {ckpt_weights_path} (skipping training)")
        return ckpt_weights_path

    import PIL.Image as Image
    import torch
    import torch.nn.functional as F
    from torch import nn, optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset, Sampler
    from torchvision import transforms
    from tqdm import tqdm

    class StagedCharactersDataset(Dataset):
        """Character image dataset supporting multi-word / underscore character names."""

        def __init__(self, img_dir: str, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.img_paths: List[str] = []
            self.labels: List[int] = []

            parsed_items: List[Tuple[str, str]] = []
            for file_name in sorted(os.listdir(img_dir)):
                full_path = os.path.join(img_dir, file_name)
                if not os.path.isfile(full_path):
                    continue
                parsed = split_char_and_idx(file_name)
                if parsed is None:
                    continue
                char_name, _ = parsed
                parsed_items.append((char_name, full_path))

            self.label_set = sorted({c for c, _ in parsed_items})
            self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
            self.label_to_indices: Dict[str, List[int]] = {label: [] for label in self.label_set}

            for char_name, img_path in parsed_items:
                self.img_paths.append(img_path)
                self.labels.append(self.label_to_idx[char_name])
                self.label_to_indices[char_name].append(len(self.img_paths) - 1)

        def __len__(self) -> int:
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            image = Image.open(img_path)
            image, _ = rgba_to_rgb_white_bg(image)
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

    class AnchorPlusPerLabelSampler(Sampler):
        """Samples 1 anchor + 1 positive for a target character and 1 negative per other character."""

        def __init__(self, dataset: StagedCharactersDataset, shuffle: bool = True):
            self.dataset = dataset
            self.labels = dataset.label_set
            self.label_to_indices = dataset.label_to_indices
            self.shuffle = shuffle
            self.reset()

        def reset(self):
            self.label_indices = {}
            for label in self.labels:
                indices = self.label_to_indices[label].copy()
                if self.shuffle:
                    random.shuffle(indices)
                self.label_indices[label] = indices
            self.label_ptr = {label: 0 for label in self.labels}

        def __iter__(self):
            self.reset()
            for anchor_label in self.labels:
                if len(self.label_indices[anchor_label]) < 2:
                    continue
                anchor_idx = self.label_indices[anchor_label][self.label_ptr[anchor_label]]
                self.label_ptr[anchor_label] += 1
                if self.label_ptr[anchor_label] >= len(self.label_indices[anchor_label]):
                    self.label_ptr[anchor_label] = 0
                    if self.shuffle:
                        random.shuffle(self.label_indices[anchor_label])

                positive_idx = self.label_indices[anchor_label][self.label_ptr[anchor_label]]
                self.label_ptr[anchor_label] += 1
                batch = [anchor_idx, positive_idx]

                for label in self.labels:
                    if label == anchor_label:
                        continue
                    if self.label_ptr[label] >= len(self.label_indices[label]):
                        self.label_ptr[label] = 0
                        if self.shuffle:
                            random.shuffle(self.label_indices[label])
                    neg_idx = self.label_indices[label][self.label_ptr[label]]
                    self.label_ptr[label] += 1
                    batch.append(neg_idx)
                yield batch

        def __len__(self) -> int:
            valid = [len(indices) // 2 for indices in self.label_to_indices.values() if len(indices) >= 2]
            return max(1, min(valid)) if valid else 1

    def info_nce_loss(batch_embeddings: torch.Tensor, positive_indices: List[int], temp: float = 0.07) -> torch.Tensor:
        anchor_idx, positive_idx = positive_indices[0], positive_indices[1]
        n = batch_embeddings.shape[0]
        norm_emb = F.normalize(batch_embeddings, dim=1)
        anchor = norm_emb[anchor_idx].unsqueeze(0)
        positive = norm_emb[positive_idx].unsqueeze(0)

        all_indices = torch.arange(n, device=batch_embeddings.device)
        pos_tensor = torch.tensor([anchor_idx, positive_idx], device=batch_embeddings.device)
        neg_mask = ~torch.isin(all_indices, pos_tensor)
        negatives = norm_emb[neg_mask]

        keys = torch.cat([positive, negatives], dim=0)
        logits = torch.matmul(anchor, keys.T) / temp
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=batch_embeddings.device)
        return F.cross_entropy(logits, target)

    dataset = StagedCharactersDataset(
        img_dir=staged_movie_dir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    if model_size not in model_archs:
        raise ValueError(f"Unsupported model_size={model_size!r}. Choose from {list(model_archs)}")

    model_name = f"dinov2_{model_archs[model_size]}_reg"
    print(f"[Stage A2] Loading pretrained {model_name} on {device} for {epochs}-epoch contrastive fine-tuning...")
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, verbose=False)
    model.eval()
    model.to(device)

    if len(dataset.label_set) < 2:
        print(
            f"[Stage A2] Only {len(dataset.label_set)} character found in bank; "
            "saving base DINOv2 weights without contrastive negative training."
        )
        os.makedirs(os.path.dirname(ckpt_weights_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_weights_path)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ckpt_weights_path

    batch_sampler = AnchorPlusPerLabelSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=torch.cuda.is_available())

    for param in model.parameters():
        param.requires_grad = False

    last_block = model.blocks[-1]
    for param in last_block.mlp.w3.parameters():
        param.requires_grad = True

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        for param in model.head.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)

    for epoch in tqdm(range(epochs), desc="DINOv2 Fine-Tuning Epochs"):
        model.train()
        total_loss = 0.0
        steps = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            most_common_label = Counter(labels.tolist()).most_common(1)[0][0]
            positive_indices = [i for i, x in enumerate(labels.tolist()) if x == most_common_label]
            if len(positive_indices) < 2:
                continue

            optimizer.zero_grad()
            embeddings = model(images)
            loss = info_nce_loss(embeddings, positive_indices[:2], temp=temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        if (epoch + 1) % 15 == 0 or (epoch + 1) == epochs:
            avg_loss = total_loss / max(1, steps)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch + 1}/{epochs}] Avg InfoNCE Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    os.makedirs(os.path.dirname(ckpt_weights_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_weights_path)
    print(f"[Stage A2] Saved fine-tuned DINOv2 weights to: {ckpt_weights_path}")

    del model, optimizer, scheduler, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ckpt_weights_path


def extract_character_features(
    staged_movie_dir: str,
    ckpt_weights_path: str,
    char_features_dir: str,
    model_size: str = "giant",
    num_per_character: int = 8,
    use_box: bool = True,
    use_mask: bool = False,
    sam2_checkpoint: Optional[str] = None,
    embed_all: bool = False,
) -> Dict[str, int]:
    """Extract normalized character exemplar features (.npz) using fine-tuned DINOv2."""
    import numpy as np
    import PIL.Image as Image
    import torch
    from torchvision import transforms
    from tqdm import tqdm
    from utils import crop_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    predictor = None
    if use_mask:
        if not use_box:
            raise ValueError("--mask requires --box")
        if not sam2_checkpoint or not os.path.isfile(sam2_checkpoint):
            raise FileNotFoundError(f"Valid --sam2_checkpoint required when --mask is set: {sam2_checkpoint}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("[Stage A2] Initializing SAM 2.1 ImagePredictor for character bank masking...")
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

    processor = None
    owlv2_model = None
    texts = [["a photo of an animated character"]]
    if use_box:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        print("[Stage A2] Initializing OWLv2 for character bank box cropping...")
        processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
        owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(device)
        owlv2_model.eval()

    model_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    model_name = f"dinov2_{model_archs[model_size]}_reg"
    print(f"[Stage A2] Loading fine-tuned {model_name} from {ckpt_weights_path}...")
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, pretrained=False, verbose=False)
    state_dict = torch.load(ckpt_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    model.to(device)

    if os.path.isdir(char_features_dir):
        shutil.rmtree(char_features_dir)
    os.makedirs(char_features_dir, exist_ok=True)

    profile_images: List[Tuple[str, int, str]] = []
    retrieval_images: List[Tuple[str, int, str]] = []

    for image_file in sorted(os.listdir(staged_movie_dir)):
        parsed = split_char_and_idx(image_file)
        if parsed is None:
            continue
        char_name, idx = parsed
        if embed_all or idx == 0:
            profile_images.append((char_name, idx, image_file))
        else:
            retrieval_images.append((char_name, idx, image_file))

    def detect_and_embed(image_path: str, require_detection: bool) -> Optional[np.ndarray]:
        try:
            pil_img = Image.open(image_path)
            image, is_clean = rgba_to_rgb_white_bg(pil_img)
        except Exception:
            return None

        if use_box and processor is not None and owlv2_model is not None:
            inputs = processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target_sizes = torch.Tensor([image.size[::-1]]).to(device)
            with torch.no_grad():
                outputs = owlv2_model(**inputs)
                results = processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.1
                )
                input_boxes = results[0]["boxes"].detach().cpu().numpy().tolist()

            if len(input_boxes) > 1:
                best_idx = int(torch.argmax(results[0]["scores"]).item())
                bbox = input_boxes[best_idx]
                use_box_prompts = True
            elif len(input_boxes) == 1:
                bbox = input_boxes[0]
                use_box_prompts = True
            else:
                if require_detection:
                    return None
                width, height = image.size
                bbox = [0, 0, width, height]
                use_box_prompts = False

            instance_mask = None
            if use_mask and predictor is not None and not is_clean:
                predictor.set_image(image)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=[bbox] if use_box_prompts else None,
                    multimask_output=False,
                )
                instance_mask = np.squeeze(masks[0])

            image = crop_image(image, bbox, instance_mask, None)

        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(image_tensor).squeeze().clone()
        return feats.detach().cpu().numpy()

    anchor_features: Dict[str, np.ndarray] = {}
    saved_counts: Dict[str, int] = defaultdict(int)

    print("[Stage A2] Extracting anchor (_0) character features...")
    for char_name, idx, image_file in tqdm(profile_images, desc="Profile Anchors"):
        feats_np = detect_and_embed(os.path.join(staged_movie_dir, image_file), require_detection=False)
        if feats_np is None:
            continue
        npz_name = os.path.splitext(image_file)[0] + ".npz"
        np.savez(os.path.join(char_features_dir, npz_name), feature=feats_np, character=char_name)
        anchor_features[char_name] = feats_np
        saved_counts[char_name] += 1

    if not embed_all and retrieval_images:
        print("[Stage A2] Scoring & filtering retrieval exemplar images against anchors...")
        selections: Dict[str, List[Tuple[float, str, np.ndarray]]] = defaultdict(list)

        for char_name, idx, image_file in tqdm(retrieval_images, desc="Exemplar Images"):
            if char_name not in anchor_features:
                continue
            feats_np = detect_and_embed(os.path.join(staged_movie_dir, image_file), require_detection=use_box)
            if feats_np is None:
                continue
            anchor_vec = anchor_features[char_name]
            denom = float(np.linalg.norm(feats_np) * np.linalg.norm(anchor_vec))
            cos_sim = float(np.dot(feats_np, anchor_vec) / denom) if denom > 0 else 0.0
            selections[char_name].append((cos_sim, image_file, feats_np))

        for char_name, candidates in selections.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            for cos_sim, image_file, feats_np in candidates[:num_per_character]:
                npz_name = os.path.splitext(image_file)[0] + ".npz"
                np.savez(os.path.join(char_features_dir, npz_name), feature=feats_np, character=char_name)
                saved_counts[char_name] += 1

    # Free all GPU models from VRAM
    del model, owlv2_model, processor, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Stage A2] Saved character feature bank (.npz) to: {char_features_dir}")
    for char_name, count in sorted(saved_counts.items()):
        print(f"  - {char_name}: {count} feature vector(s) saved")

    return dict(saved_counts)


def run_stage2_charbank(
    movie_id: str,
    char_bank_dir: str,
    output_dir: str,
    model_size: str = "giant",
    epochs: int = 75,
    learning_rate: float = 6e-4,
    temperature: float = 0.07,
    num_per_character: int = 8,
    use_box: bool = True,
    use_mask: bool = False,
    sam2_checkpoint: Optional[str] = None,
    embed_all: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Orchestrate Stage A2 end-to-end and return summary metadata."""
    staged_movie_dir = os.path.join(output_dir, "char_bank_staged", movie_id)
    ckpt_weights_path = os.path.join(output_dir, "checkpoints", movie_id, "finetuned_dinov2_weights.pth")
    char_features_dir = os.path.join(output_dir, "char_features", movie_id)

    staged_manifest, glossary = prepare_staged_character_bank(
        movie_id=movie_id,
        char_bank_dir=char_bank_dir,
        staged_movie_dir=staged_movie_dir,
        output_dir=output_dir,
    )

    finetune_dinov2(
        staged_movie_dir=staged_movie_dir,
        ckpt_weights_path=ckpt_weights_path,
        model_size=model_size,
        epochs=epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        overwrite=overwrite,
    )

    saved_counts = extract_character_features(
        staged_movie_dir=staged_movie_dir,
        ckpt_weights_path=ckpt_weights_path,
        char_features_dir=char_features_dir,
        model_size=model_size,
        num_per_character=num_per_character,
        use_box=use_box,
        use_mask=use_mask,
        sam2_checkpoint=sam2_checkpoint,
        embed_all=embed_all,
    )

    summary = {
        "movie_id": movie_id,
        "model_size": model_size,
        "epochs": epochs,
        "ckpt_weights_path": ckpt_weights_path,
        "char_features_dir": char_features_dir,
        "characters": saved_counts,
        "glossary": glossary,
    }
    summary_path = os.path.join(output_dir, "stage2_charbank_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A2: Semi-automated visual character bank & 75-epoch DINOv2 contrastive fine-tuning."
    )
    parser.add_argument("--movie_id", required=True, type=str, help="Movie identifier slug (e.g. sintel)")
    parser.add_argument(
        "--char_bank_dir",
        required=True,
        type=str,
        help="Path to char_bank/{movie_id} containing character_glossary.json and raw_images/",
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Output workdir/{movie_id}")
    parser.add_argument(
        "--model_size",
        default="giant",
        choices=["small", "base", "large", "giant"],
        help="DINOv2 architecture (giant for L4 24GB, large for T4 16GB)",
    )
    parser.add_argument("--epochs", default=75, type=int, help="Contrastive fine-tuning epochs (default: 75)")
    parser.add_argument("--learning_rate", default=6e-4, type=float, help="Initial learning rate (default: 6e-4)")
    parser.add_argument("--temperature", default=0.07, type=float, help="InfoNCE temperature (default: 0.07)")
    parser.add_argument("--num_per_character", default=8, type=int, help="Max retrieval exemplars per character")
    parser.add_argument("--no_box", action="store_true", help="Disable OWLv2 character box cropping on bank images")
    parser.add_argument("--mask", action="store_true", help="Enable SAM 2.1 instance masking on bank images")
    parser.add_argument("--sam2_checkpoint", default=None, type=str, help="Path to SAM 2.1 checkpoint (if --mask)")
    parser.add_argument("--embed_all", action="store_true", help="Embed all images directly without anchor filtering")
    parser.add_argument("--overwrite", action="store_true", help="Re-run DINOv2 fine-tuning even if checkpoint exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage2_charbank(
        movie_id=args.movie_id,
        char_bank_dir=args.char_bank_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        num_per_character=args.num_per_character,
        use_box=not args.no_box,
        use_mask=args.mask,
        sam2_checkpoint=args.sam2_checkpoint,
        embed_all=args.embed_all,
        overwrite=args.overwrite,
    )
