"""Stage A4: Audio Extraction, Subtitle Ingestion & Hybrid Speaker Attribution.

Implements Phase A Stage A4 for MovieRAG:
1. Extract 16 kHz mono movie audio (plus optional Demucs vocal stem separation).
2. Load external/embedded .srt subtitles (supporting SDH character tags) or run
   faster-whisper ASR with VAD only when no subtitle file is available.
3. Self-bootstrap an In-Movie Voice Bank using high-confidence visual tracks from
   Stage A3, optional LWTNet active-speaker lip-sync scoring, and SpeechBrain ECAPA-TDNN.
4. Attribute character speakers to all subtitle utterances using ECAPA-TDNN top-3
   cosine similarity fused with overlapping visual track / LWTNet evidence, and
   attach scene-aligned subtitles back into shots_and_scenes.json.
"""

import argparse
import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline import ensure_animated_ad_paths


def _trim_host_memory(empty_cuda: bool = False) -> None:
    """Release Python cyclic references, OS heap arenas, and optional CUDA cache."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass
    if empty_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def patch_audio_stack_compat() -> None:
    """Apply lightweight compatibility shims for newer torchaudio & huggingface_hub."""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

        # Guard torchaudio 2.8+ against missing TorchCodec when demucs/speechbrain loads or saves WAVs
        orig_load = getattr(torchaudio, "load", None)
        if orig_load is not None and not getattr(orig_load, "_movierag_patched", False):

            def _compat_load(uri, *args, **kwargs):
                try:
                    return orig_load(uri, *args, **kwargs)
                except Exception as exc:
                    if "torchcodec" not in str(exc).lower() and not str(uri).lower().endswith((".wav", ".flac", ".ogg")):
                        raise
                    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
                    tensor = torch.from_numpy(np.ascontiguousarray(data.T))
                    if kwargs.get("normalize", True) is False:
                        pass
                    return tensor, int(sr)

            _compat_load._movierag_patched = True
            torchaudio.load = _compat_load

        orig_save = getattr(torchaudio, "save", None)
        if orig_save is not None and not getattr(orig_save, "_movierag_patched", False):

            def _compat_save(uri, src, sample_rate, *args, **kwargs):
                try:
                    return orig_save(uri, src, sample_rate, *args, **kwargs)
                except Exception as exc:
                    if "torchcodec" not in str(exc).lower():
                        raise
                    arr = src.detach().cpu().numpy()
                    if arr.ndim == 2:
                        arr = arr.T
                    sf.write(str(uri), arr, int(sample_rate))

            _compat_save._movierag_patched = True
            torchaudio.save = _compat_save
    except Exception:
        pass

    try:
        import huggingface_hub
        import huggingface_hub.file_download as hf_fd

        if not getattr(hf_fd.hf_hub_download, "_movierag_patched", False):
            orig_download = hf_fd.hf_hub_download

            def _compat_hf_hub_download(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    auth_tok = kwargs.pop("use_auth_token")
                    if auth_tok is not None and "token" not in kwargs:
                        kwargs["token"] = auth_tok
                return orig_download(*args, **kwargs)

            _compat_hf_hub_download._movierag_patched = True
            hf_fd.hf_hub_download = _compat_hf_hub_download
            huggingface_hub.hf_hub_download = _compat_hf_hub_download
    except Exception:
        pass


def _resolve_scratch_dir(output_dir: str, movie_id: str) -> str:
    """Prefer fast local SSD (/content) in Colab when available; else use output_dir."""
    if os.path.isdir("/content") and os.access("/content", os.W_OK):
        scratch = os.path.join("/content", "audio_scratch", movie_id)
    else:
        scratch = os.path.join(output_dir, "audio_scratch")
    os.makedirs(scratch, exist_ok=True)
    return scratch


def _convert_to_16k_mono_wav(src_path: str, dst_wav: str) -> str:
    """Convert any audio/video stream to 16 kHz mono 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(dst_wav)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        dst_wav,
    ]
    subprocess.run(cmd, check=True)
    return dst_wav


def prepare_movie_audio(
    video_path: str,
    output_dir: str,
    movie_id: str,
    use_demucs: bool = True,
) -> Tuple[str, str]:
    """Extract 16 kHz mono WAV and optionally isolate vocals via Demucs htdemucs.

    Returns:
        (raw_wav_path, speech_wav_path) where speech_wav_path points to the isolated
        Demucs vocal track if enabled & successful, otherwise raw_wav_path.
    """
    scratch_dir = _resolve_scratch_dir(output_dir, movie_id)
    drive_audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(drive_audio_dir, exist_ok=True)

    drive_raw_wav = os.path.join(drive_audio_dir, f"{movie_id}_raw_16k.wav")
    local_raw_wav = os.path.join(scratch_dir, f"{movie_id}_raw_16k.wav")

    if os.path.isfile(drive_raw_wav) and os.path.getsize(drive_raw_wav) > 1024:
        shutil.copy2(drive_raw_wav, local_raw_wav)
        print(f"[Stage A4.1] Reusing cached 16 kHz raw audio: {drive_raw_wav}")
    else:
        print(f"[Stage A4.1] Extracting 16 kHz mono WAV from {video_path}...")
        _convert_to_16k_mono_wav(video_path, local_raw_wav)
        shutil.copy2(local_raw_wav, drive_raw_wav)

    if not use_demucs:
        return local_raw_wav, local_raw_wav

    drive_vocals_wav = os.path.join(drive_audio_dir, f"{movie_id}_vocals_16k.wav")
    local_vocals_wav = os.path.join(scratch_dir, f"{movie_id}_vocals_16k.wav")

    if os.path.isfile(drive_vocals_wav) and os.path.getsize(drive_vocals_wav) > 1024:
        shutil.copy2(drive_vocals_wav, local_vocals_wav)
        print(f"[Stage A4.1] Reusing cached Demucs vocal stem: {drive_vocals_wav}")
        return local_raw_wav, local_vocals_wav

    demucs_in_wav = os.path.join(scratch_dir, f"{movie_id}_44k_stereo.wav")
    demucs_out_dir = os.path.join(scratch_dir, "demucs_out")
    try:
        print("[Stage A4.1] Running Demucs (htdemucs --two-stems=vocals) to isolate clean speech...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-acodec",
                "pcm_s16le",
                demucs_in_wav,
            ],
            check=True,
        )
        from demucs.separate import main as demucs_main

        demucs_main(
            [
                "-n",
                "htdemucs",
                "--two-stems=vocals",
                "-o",
                demucs_out_dir,
                demucs_in_wav,
            ]
        )
        stem_name = Path(demucs_in_wav).stem
        vocals_44k = os.path.join(demucs_out_dir, "htdemucs", stem_name, "vocals.wav")
        if not os.path.isfile(vocals_44k):
            raise FileNotFoundError(f"Expected Demucs vocal stem not found: {vocals_44k}")

        _convert_to_16k_mono_wav(vocals_44k, local_vocals_wav)
        shutil.copy2(local_vocals_wav, drive_vocals_wav)
        print(f"[Stage A4.1] Saved Demucs 16 kHz vocal stem to: {drive_vocals_wav}")
        return local_raw_wav, local_vocals_wav
    except Exception as exc:
        print(f"[Stage A4.1] WARNING: Demucs vocal separation skipped/failed ({exc}); falling back to raw audio.")
        return local_raw_wav, local_raw_wav
    finally:
        if os.path.isfile(demucs_in_wav):
            try:
                os.remove(demucs_in_wav)
            except OSError:
                pass
        if os.path.isdir(demucs_out_dir):
            shutil.rmtree(demucs_out_dir, ignore_errors=True)
        _trim_host_memory(empty_cuda=True)


def _srt_ts_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS,mmm (or HH:MM:SS.mmm) into float seconds."""
    norm = ts_str.strip().replace(",", ".")
    parts = norm.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    if len(parts) == 2:
        m, s = int(parts[0]), float(parts[1])
        return m * 60.0 + s
    return float(parts[0])


def _seconds_to_srt_ts(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3600000
    m = (ms_total % 3600000) // 60000
    s = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _clean_srt_text(raw_text: str) -> str:
    """Strip HTML/ASS styling tags and normalize whitespace inside a subtitle block."""
    text = re.sub(r"<[^>]+>", "", raw_text)
    text = re.sub(r"\{\\[^}]+\}", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines).strip()


def _match_sdh_speaker(text: str, known_characters: List[str]) -> Tuple[Optional[str], str]:
    """Detect explicit SDH speaker tags like '[Sintel]: ...' or 'Shaman: ...'."""
    if not known_characters:
        return None, text

    lookup = {c.lower(): c for c in known_characters}
    patterns = [
        r"^\[([A-Za-z0-9_ \-\.]+)\]\s*:?\s*(.+)$",
        r"^\(([A-Za-z0-9_ \-\.]+)\)\s*:?\s*(.+)$",
        r"^([A-Za-z0-9_ \-]+):\s+(.+)$",
    ]
    for pat in patterns:
        match = re.match(pat, text.strip())
        if match:
            cand_raw = match.group(1).strip()
            remainder = match.group(2).strip()
            if cand_raw.lower() in lookup and remainder:
                return lookup[cand_raw.lower()], remainder
    return None, text


def parse_srt_file(srt_path: str, known_characters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Parse an .srt file into normalized utterance dicts, preserving any SDH character tags."""
    known = known_characters or []
    with open(srt_path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    blocks = re.split(r"\n{2,}", content.strip())
    utterances: List[Dict[str, Any]] = []

    ts_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        ts_line_idx = None
        ts_match = None
        for idx, ln in enumerate(lines[:3]):
            m = ts_pattern.search(ln)
            if m:
                ts_line_idx = idx
                ts_match = m
                break
        if ts_match is None or ts_line_idx is None:
            continue

        start_sec = round(_srt_ts_to_seconds(ts_match.group(1)), 3)
        end_sec = round(_srt_ts_to_seconds(ts_match.group(2)), 3)
        if end_sec <= start_sec:
            continue

        raw_dialogue = "\n".join(lines[ts_line_idx + 1 :])
        clean_text = _clean_srt_text(raw_dialogue)
        if not clean_text:
            continue

        sdh_speaker, trimmed_text = _match_sdh_speaker(clean_text, known)
        utt_idx = len(utterances)
        utterances.append(
            {
                "utt_id": f"utt_{utt_idx:04d}",
                "start_time": start_sec,
                "end_time": end_sec,
                "text": trimmed_text if sdh_speaker else clean_text,
                "speaker": sdh_speaker,
                "confidence": 1.0 if sdh_speaker else 0.0,
                "source": "sdh" if sdh_speaker else "srt",
            }
        )

    return utterances


def _find_existing_srt(video_path: str, output_dir: str, movie_id: str, subtitle_path: Optional[str]) -> Optional[str]:
    """Search explicit path, sibling video directory, and workdir for an .srt file."""
    candidates: List[str] = []
    if subtitle_path:
        candidates.append(subtitle_path)

    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_stem = Path(video_path).stem
    for d in (video_dir, output_dir):
        for stem in (movie_id, video_stem):
            candidates.append(os.path.join(d, f"{stem}.srt"))
            candidates.append(os.path.join(d, f"{stem}.SRT"))

    for cand in candidates:
        if cand and os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def _extract_embedded_srt(video_path: str, extracted_srt_path: str) -> Optional[str]:
    """Extract embedded text subtitle track from video container if one exists."""
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language",
                "-of",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            return None
        info = json.loads(probe.stdout)
        text_codecs = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text", "srt"}
        for i, stream in enumerate(info.get("streams", [])):
            if stream.get("codec_name", "").lower() in text_codecs:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        video_path,
                        "-map",
                        f"0:s:{i}",
                        extracted_srt_path,
                    ],
                    check=True,
                )
                if os.path.isfile(extracted_srt_path) and os.path.getsize(extracted_srt_path) > 16:
                    return extracted_srt_path
    except Exception:
        pass
    return None


def load_or_transcribe_subtitles(
    video_path: str,
    subtitle_path: Optional[str],
    speech_wav_path: str,
    raw_wav_path: str,
    output_dir: str,
    movie_id: str,
    known_characters: List[str],
    whisper_model_size: str = "large-v3",
    whisper_language: Optional[str] = "en",
) -> Tuple[List[Dict[str, Any]], str]:
    """Load external/embedded SRT if present, otherwise run faster-whisper ASR.

    Returns:
        (utterances, subtitle_origin_tag)
    """
    existing_srt = _find_existing_srt(video_path, output_dir, movie_id, subtitle_path)
    if existing_srt is not None:
        print(f"[Stage A4.1] Found subtitle file: {existing_srt} — skipping Whisper ASR.")
        utterances = parse_srt_file(existing_srt, known_characters=known_characters)
        print(f"[Stage A4.1] Parsed {len(utterances)} subtitle lines from {os.path.basename(existing_srt)}.")
        return utterances, f"srt:{os.path.basename(existing_srt)}"

    embedded_srt = os.path.join(output_dir, f"{movie_id}_embedded.srt")
    if _extract_embedded_srt(video_path, embedded_srt):
        print(f"[Stage A4.1] Extracted embedded text subtitle track -> {embedded_srt} (skipping Whisper ASR).")
        utterances = parse_srt_file(embedded_srt, known_characters=known_characters)
        return utterances, "embedded_srt"

    cached_whisper_srt = os.path.join(output_dir, f"{movie_id}_whisper.srt")
    if os.path.isfile(cached_whisper_srt) and os.path.getsize(cached_whisper_srt) > 0:
        print(f"[Stage A4.1] Reusing cached Whisper transcript: {cached_whisper_srt}")
        utterances = parse_srt_file(cached_whisper_srt, known_characters=known_characters)
        for u in utterances:
            if u["source"] == "srt":
                u["source"] = "faster_whisper"
        return utterances, "cached_whisper_srt"

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    asr_audio = speech_wav_path if os.path.isfile(speech_wav_path) else raw_wav_path

    print(
        f"[Stage A4.1] No .srt found. Transcribing with faster-whisper '{whisper_model_size}' "
        f"({device}/{compute_type}, language={whisper_language or 'auto'}) on {os.path.basename(asr_audio)}..."
    )
    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        asr_audio,
        language=whisper_language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
        condition_on_previous_text=True,
    )

    utterances: List[Dict[str, Any]] = []
    srt_blocks: List[str] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt or seg.end <= seg.start:
            continue
        utt_idx = len(utterances)
        start_s = round(float(seg.start), 3)
        end_s = round(float(seg.end), 3)
        utterances.append(
            {
                "utt_id": f"utt_{utt_idx:04d}",
                "start_time": start_s,
                "end_time": end_s,
                "text": txt,
                "speaker": None,
                "confidence": 0.0,
                "source": "faster_whisper",
            }
        )
        srt_blocks.append(
            f"{utt_idx + 1}\n{_seconds_to_srt_ts(start_s)} --> {_seconds_to_srt_ts(end_s)}\n{txt}\n"
        )

    with open(cached_whisper_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    print(
        f"[Stage A4.1] faster-whisper ({getattr(info, 'language', whisper_language)}) produced "
        f"{len(utterances)} utterances -> {cached_whisper_srt}"
    )
    del model
    _trim_host_memory(empty_cuda=True)
    return utterances, "faster_whisper"


def _load_wav_mono_float(wav_path: str) -> Tuple[int, Any]:
    """Load mono PCM WAV as float32 normalized to [-1.0, 1.0]."""
    import numpy as np
    from scipy.io import wavfile

    sr, wav = wavfile.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype == np.int16:
        wav_f = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav_f = wav.astype(np.float32) / 2147483648.0
    else:
        wav_f = wav.astype(np.float32)
        max_abs = float(np.max(np.abs(wav_f))) if wav_f.size else 1.0
        if max_abs > 1.5:
            wav_f = wav_f / 32768.0
    return int(sr), wav_f


def _slice_audio(wav_f, sr: int, start_sec: float, end_sec: float, min_sec: float = 0.8):
    """Slice mono float32 waveform between [start_sec, end_sec], symmetrically padded to min_sec."""
    dur_total = len(wav_f) / float(sr)
    center = 0.5 * (start_sec + end_sec)
    span = max(end_sec - start_sec, min_sec)
    s = max(0.0, min(start_sec, center - 0.5 * span))
    e = min(dur_total, max(end_sec, s + span))
    if e - s < min_sec and s > 0.0:
        s = max(0.0, e - min_sec)
    i0 = max(0, int(round(s * sr)))
    i1 = min(len(wav_f), int(round(e * sr)))
    return wav_f[i0:i1]


def _l2_normalize(vec):
    import numpy as np

    norm = float(np.linalg.norm(vec))
    return (vec / norm) if norm > 1e-12 else vec


def _encode_ecapa(classifier, audio_slice, device: str):
    """Encode a 1D float32 numpy waveform into an L2-normalized ECAPA-TDNN embedding."""
    import torch

    if audio_slice is None or len(audio_slice) < 3200:
        return None
    wav_t = torch.from_numpy(audio_slice).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = classifier.encode_batch(wav_t).squeeze().detach().cpu().numpy()
    return _l2_normalize(emb)


def _score_lwtnet_track(
    trainer,
    video_path: str,
    track: Dict[str, Any],
    shot_start_frame: int,
    shot_fps: float,
    wav_16k,
    sr: int,
    target_fps: int = 25,
) -> Optional[float]:
    """Run LWTNet_temporal_adapter lip-sync scorer on an in-memory cropped track clip."""
    import cv2
    import numpy as np
    import torch

    frame_map = track.get("track", {})
    if len(frame_map) < 15:
        return None

    sorted_items = sorted(((int(k), v) for k, v in frame_map.items()), key=lambda x: x[0])
    local_f0, local_f1 = sorted_items[0][0], sorted_items[-1][0]
    start_sec = (shot_start_frame + local_f0) / shot_fps
    end_sec = (shot_start_frame + local_f1 + 1) / shot_fps
    dur_sec = end_sec - start_sec
    if dur_sec < 0.72:
        return None

    max_w = max(int(b[2] - b[0]) for _, b in sorted_items)
    max_h = max(int(b[3] - b[1]) for _, b in sorted_items)
    if max_w < 16 or max_h < 16:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_w = min(max_w, img_w)
    max_h = min(max_h, img_h)

    crops_orig_fps: List[np.ndarray] = []
    box_by_local = {k: v for k, v in sorted_items}
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot_start_frame + local_f0)
    last_box = sorted_items[0][1]

    for lf in range(local_f0, local_f1 + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        box = box_by_local.get(lf, last_box)
        last_box = box
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        x1 = int(round(max(0, min(img_w - max_w, cx - 0.5 * max_w))))
        y1 = int(round(max(0, min(img_h - max_h, cy - 0.5 * max_h))))
        crop = frame_bgr[y1 : y1 + max_h, x1 : x1 + max_w, ::-1]
        crop_resized = cv2.resize(crop, (480, 270), interpolation=cv2.INTER_AREA)
        crops_orig_fps.append(crop_resized)
    cap.release()

    if len(crops_orig_fps) < 15:
        return None

    # Resample video crops to 25 fps as required by LWTNet's 15-frame / 60-mel stride
    n_25 = int(round(dur_sec * target_fps))
    aud_fact = int(round(sr / target_fps))
    aud_i0 = max(0, int(round(start_sec * sr)))
    aud_avail = len(wav_16k) - aud_i0
    n_25 = min(n_25, aud_avail // aud_fact)
    if n_25 < 16:
        return None

    indices = np.linspace(0, len(crops_orig_fps) - 1, n_25, dtype=int)
    vid_arr = np.stack([crops_orig_fps[i] for i in indices], axis=0).astype(np.float32)  # T H W C in 0..255
    vid_t = torch.from_numpy(vid_arr).permute(3, 0, 1, 2).unsqueeze(0)  # 1 x 3 x T x H x W

    aud_raw = wav_16k[aud_i0 : aud_i0 + n_25 * aud_fact].astype(np.float32) * 32768.0
    aud_t = torch.from_numpy(aud_raw).unsqueeze(0)  # 1 x T*640

    dataloader = [{"video": vid_t, "audio": aud_t}]
    with torch.inference_mode():
        scores = trainer.eval(dataloader)
        return float(torch.mean(scores).item())


def _maybe_load_lwtnet(lwtnet_checkpoint: Optional[str], device: str):
    """Load LWTNet_temporal_adapter and DemoEvalTrainerScoreOnly if a checkpoint path is provided."""
    if not lwtnet_checkpoint or not os.path.isfile(lwtnet_checkpoint):
        return None, None

    ensure_animated_ad_paths()
    aud_root = str(Path(__file__).resolve().parent.parent / "Animated_AD" / "character_recognition" / "audio_recognition")
    if os.path.isdir(aud_root) and aud_root not in sys.path:
        sys.path.insert(0, aud_root)

    try:
        from types import SimpleNamespace
        from model.checkpoint import load_checkpoint
        from model.lwtnet import LWTNet_temporal_adapter
        from model.sync_scorer import DemoEvalTrainerScoreOnly

        model = LWTNet_temporal_adapter().to(device)
        load_checkpoint(lwtnet_checkpoint, model)
        model.eval()
        opts = SimpleNamespace(
            gpu_id=0,
            fps=25,
            sample_rate=16000,
            n_negative_samples=0,
        )
        trainer = DemoEvalTrainerScoreOnly(model, opts)
        print(f"[Stage A4.2] Loaded LWTNet active-speaker sync model from: {lwtnet_checkpoint}")
        return model, trainer
    except Exception as exc:
        print(f"[Stage A4.2] WARNING: Could not load LWTNet checkpoint ({exc}); using solo-overlap sync fallback.")
        return None, None


def _build_track_spans(
    classified_shots: List[Dict[str, Any]],
    shots_by_idx: Dict[int, Dict[str, Any]],
    fps: float,
    min_visual_score: float = 0.45,
) -> List[Dict[str, Any]]:
    """Flatten classified shot tracks into time-indexed spans for overlap matching."""
    spans: List[Dict[str, Any]] = []
    for shot_entry in classified_shots:
        shot_idx = int(shot_entry["shot_idx"])
        shot_meta = shots_by_idx.get(shot_idx, {})
        shot_start_frame = int(shot_entry.get("start_frame", shot_meta.get("start_frame", 0)))

        for tr in shot_entry.get("tracks", []):
            score = float(tr.get("score", 0.0))
            if score < min_visual_score:
                continue
            fmap = tr.get("track", {})
            if not fmap:
                continue
            local_keys = [int(k) for k in fmap.keys()]
            g_start = shot_start_frame + min(local_keys)
            g_end = shot_start_frame + max(local_keys)
            spans.append(
                {
                    "shot_idx": shot_idx,
                    "shot_start_frame": shot_start_frame,
                    "track_id": tr.get("track_id"),
                    "label": tr.get("label"),
                    "visual_score": score,
                    "is_voice_anchor_valid": bool(tr.get("is_voice_anchor_valid", score >= 0.60)),
                    "start_frame": g_start,
                    "end_frame": g_end,
                    "start_time": round(g_start / fps, 3),
                    "end_time": round((g_end + 1) / fps, 3),
                    "track_dict": tr,
                }
            )
    return spans


def harvest_in_movie_voice_bank(
    track_spans: List[Dict[str, Any]],
    utterances: List[Dict[str, Any]],
    wav_16k,
    sr: int,
    fps: float,
    video_path: str,
    classifier,
    device: str,
    voice_bank_dir: str,
    lwtnet_trainer=None,
    voice_harvest_thresh: float = 0.60,
    lwtnet_sync_thresh: float = 0.30,
    min_examples_per_char: int = 8,
    max_examples_per_char: int = 15,
) -> Dict[str, List[Any]]:
    """Self-bootstrap ECAPA-TDNN voice exemplars from speech intervals with clear visual evidence."""
    import numpy as np

    os.makedirs(voice_bank_dir, exist_ok=True)
    voice_bank: Dict[str, List[np.ndarray]] = defaultdict(list)

    # Pass 1: Use explicit SDH subtitles if available
    for utt in utterances:
        spk = utt.get("speaker")
        if (
            utt.get("source") == "sdh"
            and spk
            and len(voice_bank[spk]) < max_examples_per_char
        ):
            sl = _slice_audio(wav_16k, sr, utt["start_time"], utt["end_time"], min_sec=0.8)
            rms = float(np.sqrt(np.mean(sl * sl))) if len(sl) else 0.0
            if rms >= 0.004:
                emb = _encode_ecapa(classifier, sl, device)
                if emb is not None:
                    voice_bank[spk].append(emb)

    # Pass 2: Match speech utterances against high-confidence visual character tracks
    # Prioritize utterances where exactly ONE speaking character is clearly visible on screen
    # (or where LWTNet lip-sync identifies the active speaker).
    candidates_by_char: Dict[str, List[Tuple[float, float, float, Dict[str, Any]]]] = defaultdict(list)

    for utt in utterances:
        if utt.get("source") == "sdh" and utt.get("speaker"):
            continue
        t0, t1 = float(utt["start_time"]), float(utt["end_time"])
        if t1 - t0 < 0.45:
            continue

        sl = _slice_audio(wav_16k, sr, t0, t1, min_sec=0.6)
        rms = float(np.sqrt(np.mean(sl * sl))) if len(sl) else 0.0
        if rms < 0.005:
            continue

        overlaps = [
            sp
            for sp in track_spans
            if min(t1, sp["end_time"]) - max(t0, sp["start_time"]) >= 0.35
        ]
        if not overlaps:
            continue

        best_per_char: Dict[str, Dict[str, Any]] = {}
        for sp in overlaps:
            char = sp["label"]
            ov_dur = min(t1, sp["end_time"]) - max(t0, sp["start_time"])
            ov_ratio = ov_dur / max(0.1, t1 - t0)
            if char not in best_per_char or sp["visual_score"] > best_per_char[char]["visual_score"]:
                best_per_char[char] = {**sp, "ov_ratio": ov_ratio}

        # Filter to confident candidates (>= 0.60 preferred, else dominant >= 70% overlap)
        strong_chars = [c for c, info in best_per_char.items() if info["visual_score"] >= voice_harvest_thresh]
        solo_char: Optional[str] = None
        if len(strong_chars) == 1:
            solo_char = strong_chars[0]
        elif len(strong_chars) >= 2:
            ranked_ov = sorted(
                ((best_per_char[c]["ov_ratio"], best_per_char[c]["visual_score"], c) for c in strong_chars),
                reverse=True,
            )
            if ranked_ov[0][0] >= 0.70 and ranked_ov[1][0] < 0.30:
                solo_char = ranked_ov[0][2]

        if solo_char is not None:
            info = best_per_char[solo_char]
            sync_score = 1.0
            if lwtnet_trainer is not None:
                s_lwt = _score_lwtnet_track(
                    lwtnet_trainer,
                    video_path,
                    info["track_dict"],
                    info["shot_start_frame"],
                    fps,
                    wav_16k,
                    sr,
                )
                if s_lwt is not None:
                    sync_score = s_lwt
                    if sync_score < lwtnet_sync_thresh:
                        continue
            priority = info["visual_score"] * info["ov_ratio"] * sync_score
            candidates_by_char[solo_char].append((priority, t0, t1, info["track_dict"]))

        elif len(best_per_char) >= 2 and lwtnet_trainer is not None:
            scored_chars: List[Tuple[float, str, Dict[str, Any]]] = []
            for c, info in best_per_char.items():
                if info["visual_score"] < 0.50:
                    continue
                s_lwt = _score_lwtnet_track(
                    lwtnet_trainer,
                    video_path,
                    info["track_dict"],
                    info["shot_start_frame"],
                    fps,
                    wav_16k,
                    sr,
                )
                if s_lwt is not None and s_lwt >= lwtnet_sync_thresh:
                    scored_chars.append((s_lwt * info["visual_score"], c, info))
            if scored_chars:
                scored_chars.sort(key=lambda x: x[0], reverse=True)
                best_comb, best_c, best_info = scored_chars[0]
                margin_ok = len(scored_chars) == 1 or (best_comb - scored_chars[1][0]) >= 0.05
                if margin_ok:
                    candidates_by_char[best_c].append((best_comb, t0, t1, best_info["track_dict"]))

    # Secondary top-up if a speaker only appeared alongside another non-speaking character
    # with weaker visual score (< voice_harvest_thresh)
    for char, cand_list in list(candidates_by_char.items()):
        cand_list.sort(key=lambda x: x[0], reverse=True)
        for _, t0, t1, _ in cand_list:
            if len(voice_bank[char]) >= max_examples_per_char:
                break
            sl = _slice_audio(wav_16k, sr, t0, t1, min_sec=0.8)
            emb = _encode_ecapa(classifier, sl, device)
            if emb is not None:
                voice_bank[char].append(emb)

    # Save harvested .npz exemplars
    saved_summary: Dict[str, int] = {}
    for char, embs in sorted(voice_bank.items()):
        for idx, emb in enumerate(embs):
            npz_path = os.path.join(voice_bank_dir, f"{char}_{idx}.npz")
            np.savez(npz_path, feature=emb, character=char)
        saved_summary[char] = len(embs)

    print(f"[Stage A4.2] Self-bootstrapped in-movie voice bank: {saved_summary}")
    return dict(voice_bank)


def _top3_cosine_scores(query_emb, voice_bank: Dict[str, List[Any]]) -> Dict[str, float]:
    """Compute top-3 average cosine similarity against each character's voice bank."""
    import numpy as np

    scores: Dict[str, float] = {}
    if query_emb is None:
        return scores
    for char, embs in voice_bank.items():
        if not embs:
            continue
        sims = [float(np.dot(query_emb, b_emb)) for b_emb in embs]
        sims.sort(reverse=True)
        k = min(3, len(sims))
        scores[char] = sum(sims[:k]) / float(k)
    return scores


def attribute_speakers_to_subtitles(
    utterances: List[Dict[str, Any]],
    track_spans: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    wav_16k,
    sr: int,
    fps: float,
    video_path: str,
    classifier,
    device: str,
    voice_bank: Dict[str, List[Any]],
    lwtnet_trainer=None,
    audio_threshold: float = 0.40,
    visual_alpha: float = 0.85,
) -> List[Dict[str, Any]]:
    """Assign speaker character names to subtitles using ECAPA-TDNN + visual enhancement."""
    attributed: List[Dict[str, Any]] = []

    for utt in utterances:
        item = dict(utt)
        if item.get("source") == "sdh" and item.get("speaker"):
            attributed.append(item)
            continue

        t0, t1 = float(item["start_time"]), float(item["end_time"])
        mid_t = 0.5 * (t0 + t1)

        sl = _slice_audio(wav_16k, sr, t0, t1, min_sec=0.8)
        query_emb = _encode_ecapa(classifier, sl, device)
        audio_scores = _top3_cosine_scores(query_emb, voice_bank)

        best_audio_char: Optional[str] = None
        best_audio_score = 0.0
        if audio_scores:
            best_audio_char, best_audio_score = max(audio_scores.items(), key=lambda x: x[1])

        # Overlapping visual character candidates during [t0, t1]
        vis_candidates: Dict[str, Dict[str, Any]] = {}
        for sp in track_spans:
            char = sp["label"]
            ov = min(t1, sp["end_time"]) - max(t0, sp["start_time"])
            if ov <= 0.15:
                continue
            if char not in vis_candidates or sp["visual_score"] > vis_candidates[char]["visual_score"]:
                vis_candidates[char] = sp

        # Parent scene characters (used when camera cuts away cleanly during a line)
        scene_chars: Dict[str, float] = {}
        for sc in scenes:
            if float(sc["start_time"]) <= mid_t <= float(sc["end_time"]):
                for cp in sc.get("characters_present", []):
                    cname = cp["name"]
                    scene_chars[cname] = max(scene_chars.get(cname, 0.0), float(cp.get("max_score", 0.0)))
                break

        assigned_speaker: Optional[str] = None
        assigned_conf: float = 0.0
        source_tag: str = "unattributed"

        if best_audio_char is not None and best_audio_score >= audio_threshold:
            # High-confidence vocal match
            assigned_speaker = best_audio_char
            assigned_conf = best_audio_score
            source_tag = "ecapa_voice_bank"
        else:
            # Audio confidence was below threshold: apply visual enhancement & fusion
            fused_scores: Dict[str, Tuple[float, str]] = {}
            for char, sp in vis_candidates.items():
                aud_s = max(0.0, audio_scores.get(char, 0.0))
                vis_s = float(sp["visual_score"])
                sync_s = 1.0
                src = "av_visual_fallback"

                if lwtnet_trainer is not None:
                    lwt_val = _score_lwtnet_track(
                        lwtnet_trainer,
                        video_path,
                        sp["track_dict"],
                        sp["shot_start_frame"],
                        fps,
                        wav_16k,
                        sr,
                    )
                    if lwt_val is not None:
                        sync_s = lwt_val
                        src = "avsync_enhanced"
                        vis_comb = sync_s * vis_s * visual_alpha
                        score = max(vis_comb, 0.45 * aud_s + 0.55 * vis_comb)
                    else:
                        score = 0.50 * aud_s + 0.50 * vis_s * 0.75
                else:
                    # Without LWTNet weights: combine vocal match + solo on-screen bonus
                    solo_bonus = 0.12 if len(vis_candidates) == 1 else 0.0
                    score = 0.45 * aud_s + 0.55 * (vis_s * 0.65) + solo_bonus
                    src = "av_solo_enhanced" if len(vis_candidates) == 1 else "av_fused"

                if char not in fused_scores or score > fused_scores[char][0]:
                    fused_scores[char] = (score, src)

            if fused_scores:
                best_char, (best_fused, best_src) = max(fused_scores.items(), key=lambda x: x[1][0])
                if best_audio_char and best_audio_score > best_fused:
                    assigned_speaker = best_audio_char
                    assigned_conf = best_audio_score
                    source_tag = "ecapa_low_conf"
                elif best_fused >= 0.30:
                    assigned_speaker = best_char
                    assigned_conf = best_fused
                    source_tag = best_src
            elif best_audio_char is not None and best_audio_score >= 0.25:
                # Off-screen line with mild vocal similarity
                assigned_speaker = best_audio_char
                assigned_conf = best_audio_score
                source_tag = "ecapa_offscreen"
            elif len(scene_chars) == 1:
                only_char, sc_score = next(iter(scene_chars.items()))
                if sc_score >= 0.55:
                    assigned_speaker = only_char
                    assigned_conf = sc_score * 0.55
                    source_tag = "scene_solo_fallback"

        item["speaker"] = assigned_speaker or "Unknown"
        item["confidence"] = round(float(assigned_conf), 4)
        item["source"] = source_tag if assigned_speaker else "unattributed"
        attributed.append(item)

    return attributed


def attach_subtitles_to_scenes(
    shots_and_scenes: Dict[str, Any],
    attributed_subtitles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach each utterance to its matching Scene Window in shots_and_scenes.json."""
    scenes = shots_and_scenes.get("scenes", [])
    for sc in scenes:
        sc["subtitles"] = []

    for utt in attributed_subtitles:
        mid_t = 0.5 * (float(utt["start_time"]) + float(utt["end_time"]))
        placed = False
        for sc in scenes:
            if float(sc["start_time"]) <= mid_t <= float(sc["end_time"]):
                sc["subtitles"].append(utt)
                placed = True
                break
        if not placed and scenes:
            # Clamp slightly out-of-bounds subtitles to nearest scene midpoint
            nearest = min(
                scenes,
                key=lambda s: abs(0.5 * (float(s["start_time"]) + float(s["end_time"])) - mid_t),
            )
            nearest["subtitles"].append(utt)

    return shots_and_scenes


def run_stage4_subtitles(
    video_path: str,
    output_dir: str,
    subtitle_path: Optional[str] = None,
    lwtnet_checkpoint: Optional[str] = None,
    use_demucs: bool = True,
    whisper_model_size: str = "large-v3",
    whisper_language: Optional[str] = "en",
    audio_threshold: float = 0.40,
    voice_harvest_thresh: float = 0.60,
    visual_alpha: float = 0.85,
) -> Dict[str, Any]:
    """Orchestrate Stage A4 end-to-end and persist attributed subtitles + updated scenes."""
    import torch

    patch_audio_stack_compat()
    from speechbrain.inference.speaker import EncoderClassifier

    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    tracks_file = os.path.join(output_dir, "classified_tracks.json")
    charbank_summary_file = os.path.join(output_dir, "stage2_charbank_summary.json")

    if not os.path.isfile(shots_file):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_file}")
    if not os.path.isfile(tracks_file):
        raise FileNotFoundError(f"classified_tracks.json not found: {tracks_file}")

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)
    with open(tracks_file, "r", encoding="utf-8") as f:
        classified_shots = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    fps = float(shots_and_scenes.get("fps") or 24.0)
    shots_by_idx = {int(s["shot_idx"]): s for s in shots_and_scenes.get("shots", [])}

    known_characters: List[str] = []
    if os.path.isfile(charbank_summary_file):
        with open(charbank_summary_file, "r", encoding="utf-8") as f:
            cb_summary = json.load(f)
        glossary = cb_summary.get("glossary", {}) or {}
        known_characters = sorted(glossary.keys() or cb_summary.get("characters", {}).keys())
    if not known_characters:
        known_characters = sorted(
            {tr.get("label") for sh in classified_shots for tr in sh.get("tracks", []) if tr.get("label")}
        )

    # 1. Extract 16 kHz audio & optional Demucs vocal stem
    raw_wav, speech_wav = prepare_movie_audio(
        video_path=video_path,
        output_dir=output_dir,
        movie_id=movie_id,
        use_demucs=use_demucs,
    )

    # 2. Load SRT/embedded subtitles or run faster-whisper fallback
    utterances, subtitle_origin = load_or_transcribe_subtitles(
        video_path=video_path,
        subtitle_path=subtitle_path,
        speech_wav_path=speech_wav,
        raw_wav_path=raw_wav,
        output_dir=output_dir,
        movie_id=movie_id,
        known_characters=known_characters,
        whisper_model_size=whisper_model_size,
        whisper_language=whisper_language,
    )

    # 3. Load speech audio + ECAPA-TDNN + optional LWTNet
    sr, wav_16k = _load_wav_mono_float(speech_wav)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa_savedir = os.path.join(
        os.environ.get("HF_HOME", os.path.join(output_dir, "cache")),
        "speechbrain_ecapa",
    )
    os.makedirs(ecapa_savedir, exist_ok=True)
    print(f"[Stage A4.2] Loading SpeechBrain ECAPA-TDNN on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ecapa_savedir,
        run_opts={"device": device},
    )

    lwtnet_model, lwtnet_trainer = _maybe_load_lwtnet(lwtnet_checkpoint, device=device)

    track_spans = _build_track_spans(
        classified_shots=classified_shots,
        shots_by_idx=shots_by_idx,
        fps=fps,
        min_visual_score=0.45,
    )

    voice_bank_dir = os.path.join(output_dir, "voice_bank", movie_id)
    voice_bank = harvest_in_movie_voice_bank(
        track_spans=track_spans,
        utterances=utterances,
        wav_16k=wav_16k,
        sr=sr,
        fps=fps,
        video_path=video_path,
        classifier=classifier,
        device=device,
        voice_bank_dir=voice_bank_dir,
        lwtnet_trainer=lwtnet_trainer,
        voice_harvest_thresh=voice_harvest_thresh,
    )

    # 4. Hybrid speaker attribution
    attributed_subtitles = attribute_speakers_to_subtitles(
        utterances=utterances,
        track_spans=track_spans,
        scenes=shots_and_scenes.get("scenes", []),
        wav_16k=wav_16k,
        sr=sr,
        fps=fps,
        video_path=video_path,
        classifier=classifier,
        device=device,
        voice_bank=voice_bank,
        lwtnet_trainer=lwtnet_trainer,
        audio_threshold=audio_threshold,
        visual_alpha=visual_alpha,
    )

    # 5. Update shots_and_scenes.json and save summary artifacts
    shots_and_scenes = attach_subtitles_to_scenes(shots_and_scenes, attributed_subtitles)
    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots_and_scenes, f, indent=2)

    subtitles_out_path = os.path.join(output_dir, "attributed_subtitles.json")
    with open(subtitles_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "movie_id": movie_id,
                "subtitle_origin": subtitle_origin,
                "used_demucs": bool(use_demucs and speech_wav != raw_wav),
                "used_lwtnet": lwtnet_trainer is not None,
                "total_utterances": len(attributed_subtitles),
                "utterances": attributed_subtitles,
            },
            f,
            indent=2,
        )

    speaker_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    for u in attributed_subtitles:
        speaker_counts[u["speaker"]] += 1
        source_counts[u["source"]] += 1

    summary = {
        "movie_id": movie_id,
        "subtitle_origin": subtitle_origin,
        "used_demucs": bool(use_demucs and speech_wav != raw_wav),
        "used_lwtnet": lwtnet_trainer is not None,
        "voice_bank_counts": {k: len(v) for k, v in voice_bank.items()},
        "total_utterances": len(attributed_subtitles),
        "speaker_counts": dict(speaker_counts),
        "source_counts": dict(source_counts),
        "attributed_subtitles_path": subtitles_out_path,
    }
    summary_path = os.path.join(output_dir, "stage4_audio_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del classifier, lwtnet_model, lwtnet_trainer, wav_16k
    _trim_host_memory(empty_cuda=True)

    print(f"[Stage A4.3] Saved attributed subtitles -> {subtitles_out_path}")
    print(f"[Stage A4.3] Speaker distribution : {dict(speaker_counts)}")
    print(f"[Stage A4.3] Attribution sources  : {dict(source_counts)}")
    return summary


def apply_subtitle_speaker_fixes(
    output_dir: str,
    speaker_overrides: Dict[str, str],
) -> Dict[str, Any]:
    """Manually patch speaker labels in attributed_subtitles.json & shots_and_scenes.json.

    Args:
        output_dir: Workdir containing attributed_subtitles.json and shots_and_scenes.json.
        speaker_overrides: Dict mapping utt_id (e.g. 'utt_0009'), start_time string
            (e.g. '143.22'), or exact subtitle text to the corrected speaker name.
    """
    subtitles_path = os.path.join(output_dir, "attributed_subtitles.json")
    shots_path = os.path.join(output_dir, "shots_and_scenes.json")
    summary_path = os.path.join(output_dir, "stage4_audio_summary.json")

    if not os.path.isfile(subtitles_path):
        raise FileNotFoundError(f"attributed_subtitles.json not found: {subtitles_path}")
    if not os.path.isfile(shots_path):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_path}")

    with open(subtitles_path, "r", encoding="utf-8") as f:
        sub_payload = json.load(f)
    with open(shots_path, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    utterances = sub_payload.get("utterances", [])
    overrides = speaker_overrides or {}
    changes = 0

    for idx, utt in enumerate(utterances):
        orig_speaker = utt.get("speaker")
        new_speaker: Optional[str] = None
        utt_id = str(utt.get("utt_id") or f"utt_{idx:04d}")
        utt_text = str(utt.get("text", "")).strip()
        t_start = float(utt.get("start_time", 0.0))
        t_str_2 = f"{t_start:.2f}"

        # Pure manual overrides: match by utt_id, 2-decimal start_time, or exact text
        if utt_id in overrides:
            new_speaker = overrides[utt_id]
        elif str(idx) in overrides:
            new_speaker = overrides[str(idx)]
        elif t_str_2 in overrides:
            new_speaker = overrides[t_str_2]
        elif utt_text in overrides:
            new_speaker = overrides[utt_text]

        if new_speaker and new_speaker != orig_speaker:
            utt["speaker"] = new_speaker
            utt["confidence"] = 1.0
            utt["source"] = "manual_override"
            changes += 1

    with open(subtitles_path, "w", encoding="utf-8") as f:
        json.dump(sub_payload, f, indent=2)

    shots_and_scenes = attach_subtitles_to_scenes(shots_and_scenes, utterances)
    with open(shots_path, "w", encoding="utf-8") as f:
        json.dump(shots_and_scenes, f, indent=2)

    speaker_counts: Dict[str, int] = defaultdict(int)
    for u in utterances:
        speaker_counts[u.get("speaker", "Unknown")] += 1

    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s4 = json.load(f)
        s4["speaker_counts"] = dict(speaker_counts)
        s4["manual_speaker_fixes_applied"] = changes
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(s4, f, indent=2)

    print(f"[Stage A4 Fix] Updated {changes} utterance speaker label(s) -> {dict(speaker_counts)}")
    return {"updated_count": changes, "speaker_counts": dict(speaker_counts), "utterances": utterances}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A4: Audio & Hybrid Speaker Diarization")
    parser.add_argument("--video_path", required=True, help="Input movie video path (.mp4/.mkv)")
    parser.add_argument("--output_dir", required=True, help="Workdir containing shots_and_scenes.json")
    parser.add_argument("--subtitle_path", default=None, help="Optional .srt file path (skips Whisper)")
    parser.add_argument("--lwtnet_checkpoint", default=None, help="Optional LWTNet checkpoint path")
    parser.add_argument("--no_demucs", action="store_true", help="Disable Demucs vocal stem separation")
    parser.add_argument("--whisper_model", default="large-v3", help="faster-whisper model size fallback")
    parser.add_argument("--whisper_language", default="en", help="Language code for faster-whisper")
    parser.add_argument("--audio_threshold", type=float, default=0.40, help="ECAPA cosine threshold")
    parser.add_argument("--voice_harvest_thresh", type=float, default=0.60, help="Visual anchor threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage4_subtitles(
        video_path=args.video_path,
        output_dir=args.output_dir,
        subtitle_path=args.subtitle_path,
        lwtnet_checkpoint=args.lwtnet_checkpoint,
        use_demucs=not args.no_demucs,
        whisper_model_size=args.whisper_model,
        whisper_language=args.whisper_language,
        audio_threshold=args.audio_threshold,
        voice_harvest_thresh=args.voice_harvest_thresh,
    )
