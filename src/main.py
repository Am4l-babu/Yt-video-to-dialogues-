"""
YouTube Dialogue Clipper
========================
Downloads a YouTube video or a whole playlist and saves every dialogue as its own
audio clip, one folder per video.

Two ways of finding and naming the dialogues:
  1. Title cards (preferred): videos that show a card like "3 / Ithokke enthu...."
     while each dialogue plays. The card's number and name become the filename
     (03_Ithokke_enthu.wav) and the card's on-screen time gives the exact cut points.
  2. Fallback: split on silence and name each clip from its first words, transcribed
     on the GPU with faster-whisper.

Usage (from the project root):
    python src/main.py                                  # uses YOUTUBE_URL below
    python src/main.py "https://youtube.com/playlist?list=..."
    python src/main.py "https://youtu.be/VIDEO_ID"
    python src/main.py --file temp_downloads/abc.wav    # reprocess a local audio file
"""

from __future__ import annotations

import argparse
import difflib
import gc
import os
import re
import shutil
import site
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
YOUTUBE_URL = "https://youtube.com/playlist?list=PLVQFzVxw8nVJtakfVcyfTnVavK_HvJUql"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "temp_downloads"
OUTPUT_DIR = PROJECT_ROOT / "output_clips"
SKIP_COMPLETED_VIDEOS = True    # re-running a playlist skips videos whose folder already exists
DOWNLOAD_ATTEMPTS = 3           # retries for YouTube's occasional one-off 403 errors

# --- Title-card detection ---
USE_TITLE_CARDS = True          # False = always use silence split + Whisper naming
CARD_SCAN_FPS = 4               # frames scanned per second; 4 = cut points accurate to 0.25 s
CARD_VIDEO_MAX_HEIGHT = 360     # low-res video is enough for OCR and downloads fast
# A card is a frame that is mostly one plain background colour (any colour: the channel
# uses yellow and orange cards), with a number line ("3" or "3.") above the dialogue name.
CARD_MIN_BG_FRACTION = 0.7      # share of the frame that must be the background colour
BOUNDARY_SNAP_MS = 500          # cut at the quietest point within +/- this of each card change
OCR_ON_GPU = True               # run card OCR on the RTX 4060 (needs onnxruntime-gpu)

# --- Silence splitting (fallback mode, and trimming of card clips) ---
MIN_SILENCE_LEN_MS = 600    # a pause must last this long to count as a split point
SILENCE_THRESH_DB = -40     # anything quieter than this (dBFS) counts as silence
KEEP_SILENCE_MS = 150       # padding kept on both ends so words aren't clipped
SEEK_STEP_MS = 10           # silence-scan resolution; 1 = exact but ~10x slower
MIN_CHUNK_LEN_MS = 1500     # fallback mode: discard chunks shorter than this

# --- Whisper, fallback mode only (RTX 4060 8 GB, fp16: "small" ~1 GB, "large-v3" ~4 GB VRAM) ---
# Malayalam is a low-resource language for Whisper: "small"/"medium" produce mostly
# garbage for it, so "large-v3" is the smallest model that gives usable names.
WHISPER_MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"        # uses Tensor cores on the RTX 4060
LANGUAGE: str | None = "ml"     # Malayalam; None = auto-detect per clip (unreliable on short clips)
BEAM_SIZE = 5                   # 1 = fastest (greedy), 5 = more accurate
ALLOW_CPU_FALLBACK = True       # fall back to CPU if the GPU cannot load the model

# --- File naming ---
MAX_WORDS_IN_NAME = 5
MAX_FILENAME_LEN = 50

# --- Export ---
EXPORT_FORMAT = "wav"           # "wav" or "mp3"
MP3_BITRATE = "320k"
DELETE_RAW_AFTER_SUCCESS = True

# =============================================================================
# WINDOWS CUDA DLL SETUP (must run before importing faster_whisper)
# =============================================================================
_dll_handles: list = []


def _register_nvidia_dlls() -> None:
    """Expose pip-installed cuBLAS/cuDNN DLLs (site-packages/nvidia/*/bin) to CTranslate2.

    Windows does not search these folders by default, which otherwise causes
    'Could not locate cudnn_ops64_9.dll' / 'cublas64_12.dll not found' crashes.
    """
    if os.name != "nt":
        return
    search_roots = {*site.getsitepackages(), site.getusersitepackages()}
    for root in search_roots:
        nvidia_dir = Path(root) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            try:
                _dll_handles.append(os.add_dll_directory(str(bin_dir)))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            except OSError:
                pass


_register_nvidia_dlls()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import ctranslate2  # noqa: E402  (installed with faster-whisper)
    import numpy as np  # noqa: E402
    import onnxruntime  # noqa: E402  (onnxruntime-gpu, used by RapidOCR)
    import yt_dlp  # noqa: E402
    from faster_whisper import WhisperModel  # noqa: E402
    from pydub import AudioSegment  # noqa: E402
    from pydub.exceptions import CouldntDecodeError, CouldntEncodeError  # noqa: E402
    from pydub.silence import detect_leading_silence, split_on_silence  # noqa: E402
    from rapidocr import RapidOCR  # noqa: E402
    from tqdm import tqdm  # noqa: E402
except ImportError as _exc:
    sys.exit(f"Missing dependency '{_exc.name}'. Run: pip install -r requirements.txt")


# =============================================================================
# DATA TYPES, ERRORS & HELPERS
# =============================================================================
class PipelineError(Exception):
    """A user-facing error that stops processing of the current video."""


@dataclass
class VideoJob:
    index: int
    video_id: str
    title: str
    url: str
    local_path: Path | None = None  # set when processing a local file instead of YouTube

    @property
    def folder_name(self) -> str:
        if self.local_path:
            return sanitize_filename(self.local_path.stem) or "local_file"
        return f"{self.index:02d}_{self.video_id}"


@dataclass
class VideoResult:
    job: VideoJob
    created: int = 0
    failed: int = 0
    status: str = ""
    ok: bool = True


@dataclass
class TitleCard:
    number: int
    name: str
    start: float  # seconds
    end: float


@dataclass
class _Shot:
    start: float
    end: float
    frame: np.ndarray


_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Devanagari letters whose +0x400 slot is unassigned or means something else in Malayalam.
_DEVANAGARI_NO_MATCH = {
    "़": "",        # nukta (+0x400 would be the unrelated vertical-bar virama)
    "ॅ": "െ",  # candra e sign -> e sign
    "ॉ": "ൊ",  # candra o sign -> o sign
    "ऍ": "എ",  # candra E -> E
    "ऑ": "ഒ",  # candra O -> O
}


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _is_gpu_problem(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cudnn", "cublas", "out of memory", "gpu", "float16"))


def ensure_directories() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise PipelineError(
            f"{', '.join(missing)} not found on PATH. Install FFmpeg "
            "(e.g. `winget install Gyan.FFmpeg`) and restart VS Code."
        )


def sanitize_filename(text: str) -> str:
    """Turn text into a safe Windows filename stem (may return '')."""
    name = _ILLEGAL_CHARS.sub("", text)
    # Cosmetic: drop punctuation/symbols (commas, periods, ...) but keep combining marks,
    # which non-Latin scripts (Malayalam, Hindi, Tamil, ...) need for vowel signs.
    name = "".join(
        ch for ch in name
        if ch in "'-" or not unicodedata.category(ch).startswith(("P", "S"))
    )
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    name = name[:MAX_FILENAME_LEN].strip(" ._")  # Windows forbids trailing dots/spaces
    if name.upper() in _RESERVED_NAMES:
        name = f"{name}_clip"
    return name


def unique_path(directory: Path, stem: str, ext: str) -> Path:
    """Return directory/stem.ext, or stem_1.ext, stem_2.ext, ... if taken."""
    candidate = directory / f"{stem}.{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}.{ext}"
        counter += 1
    return candidate


def devanagari_to_malayalam(text: str) -> str:
    """Whisper often writes Malayalam speech in Devanagari letters. Both scripts share the
    same ISCII-derived Unicode layout, offset by 0x400, so map each letter across."""
    out = []
    for ch in text:
        if ch in _DEVANAGARI_NO_MATCH:
            out.append(_DEVANAGARI_NO_MATCH[ch])
        elif "ऀ" <= ch <= "ॿ":
            target = chr(ord(ch) + 0x400)
            if unicodedata.name(target, "").startswith("MALAYALAM"):
                out.append(target)
            # else: no Malayalam counterpart (danda, etc.), drop it
        else:
            out.append(ch)
    return "".join(out)


# =============================================================================
# STEP 1: LIST & DOWNLOAD
# =============================================================================
class _DownloadProgress:
    """yt-dlp progress hook that drives a tqdm bar."""

    def __init__(self, desc: str) -> None:
        self.desc = desc
        self.bar: tqdm | None = None

    def __call__(self, d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if self.bar is None:
                self.bar = tqdm(
                    total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=self.desc, dynamic_ncols=True, leave=False,
                )
            elif total and self.bar.total != total:
                self.bar.total = total
            self.bar.update(d.get("downloaded_bytes", 0) - self.bar.n)
        elif status in ("finished", "error"):
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def list_videos(url: str) -> list[VideoJob]:
    """Expand a playlist URL into its videos; a single-video URL gives one job."""
    opts = {"quiet": True, "extract_flat": "in_playlist", "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.YoutubeDLError as exc:
        raise PipelineError(
            f"Could not read the URL. Check it and your internet connection.\n  Details: {exc}"
        ) from exc
    if not info:
        raise PipelineError("yt-dlp returned no information for this URL.")

    if info.get("entries") is None:
        video_id = info["id"]
        return [VideoJob(1, video_id, info.get("title") or video_id,
                         info.get("webpage_url") or url)]

    jobs = []
    for position, entry in enumerate(info["entries"], start=1):
        if not entry or not entry.get("id"):
            continue
        video_id = entry["id"]
        jobs.append(VideoJob(
            index=entry.get("playlist_index") or position,
            video_id=video_id,
            title=entry.get("title") or video_id,
            url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
        ))
    return jobs


def _download(url: str, fmt: str, outtmpl: str, desc: str, postprocessors: list | None = None) -> Path:
    progress = _DownloadProgress(desc)
    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "progress_hooks": [progress],
        "postprocessors": postprocessors or [],
        "overwrites": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
    }
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.YoutubeDLError as exc:
            # YouTube sometimes answers 403 for one request and fine for the next.
            if attempt == DOWNLOAD_ATTEMPTS:
                raise PipelineError(
                    f"Download failed. Check the URL and your internet connection.\n  Details: {exc}"
                ) from exc
            tqdm.write(f"  {desc} download failed (attempt {attempt}/{DOWNLOAD_ATTEMPTS}); retrying...")
            time.sleep(3 * attempt)
        finally:
            progress.close()

    requested = (info or {}).get("requested_downloads") or []
    if not requested or not requested[0].get("filepath"):
        raise PipelineError(f"yt-dlp did not report a downloaded file for {url}")
    path = Path(requested[0]["filepath"])
    if not path.exists():
        raise PipelineError(f"Expected file was not created: {path}")
    return path


def download_audio(job: VideoJob) -> Path:
    """Best available audio stream, converted to WAV."""
    return _download(
        job.url, "bestaudio/best", str(TEMP_DIR / f"{job.video_id}.%(ext)s"), "Audio",
        postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    )


def download_card_video(job: VideoJob) -> Path:
    """Small video-only stream, used just to read the title cards."""
    h = CARD_VIDEO_MAX_HEIGHT
    fmt = f"bv[height<={h}]/bv*[height<={h}]/b[height<={h}]/wv*/b"
    return _download(job.url, fmt, str(TEMP_DIR / f"{job.video_id}_video.%(ext)s"), "Video")


def load_audio(path: Path) -> AudioSegment:
    try:
        return AudioSegment.from_file(path)
    except FileNotFoundError as exc:
        raise PipelineError("FFmpeg could not be launched by pydub. Is FFmpeg on PATH?") from exc
    except CouldntDecodeError as exc:
        raise PipelineError(f"Could not decode audio file: {path}") from exc


# =============================================================================
# STEP 2a: TITLE CARDS -> CLIPS
# =============================================================================
_SCAN_W, _SCAN_H = 640, 360


def _iter_frames(video_path: Path):
    """Yield (seconds, RGB frame) at CARD_SCAN_FPS, decoded by FFmpeg straight into memory."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"fps={CARD_SCAN_FPS},scale={_SCAN_W}:{_SCAN_H}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = _SCAN_W * _SCAN_H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        index = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield index / CARD_SCAN_FPS, np.frombuffer(buf, np.uint8).reshape(_SCAN_H, _SCAN_W, 3)
            index += 1
    finally:
        proc.kill()
        proc.wait()


def _split_into_shots(video_path: Path, duration: float | None) -> list[_Shot]:
    """Group frames into shots: a new shot starts whenever a visible part of the frame changes.

    Only one frame per shot is OCR'd later, which is what keeps card detection fast.
    """
    shots: list[_Shot] = []
    prev = None
    step = 1 / CARD_SCAN_FPS
    total = int(duration * CARD_SCAN_FPS) if duration else None
    for t, frame in tqdm(_iter_frames(video_path), total=total, desc="Scanning video",
                         unit="frame", dynamic_ncols=True, leave=False):
        small = frame[::4, ::4].astype(np.int16)
        # Fraction of pixels that changed sharply: catches a text swap on a still background,
        # which barely moves the frame's average difference.
        changed = 1.0 if prev is None else (np.abs(small - prev).max(axis=-1) > 40).mean()
        if changed > 0.002:
            shots.append(_Shot(t, t + step, frame))
        else:
            shots[-1].end = t + step
            if t - shots[-1].start <= 0.75:  # use a frame ~0.75 s in, after entry animations
                shots[-1].frame = frame
        prev = small
    return shots


def _card_background(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """The frame's dominant colour and the share of pixels close to it."""
    pixels = frame[::4, ::4].reshape(-1, 3).astype(np.int16)
    bg = np.median(pixels, axis=0).astype(np.int16)
    fraction = float((np.abs(pixels - bg).sum(axis=1) < 60).mean())
    return bg, fraction


_NUMBER_LINE = re.compile(r"(\d{1,3})\s*[.:)]?")


def _norm_text(text: str) -> str:
    return re.sub(r"[\W_]", "", text).lower()


def _same_name(a: str, b: str) -> bool:
    """OCR of the same card can differ slightly between frames; treat near-matches as equal."""
    return difflib.SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio() > 0.75


def _find_watermarks(reads: list) -> set[str]:
    """Text seen across most of the video (a channel watermark) rather than on one card.

    A card name only appears while its card is shown, so text that keeps turning up over
    more than half of the card section is treated as a watermark.
    """
    if len(reads) < 3:
        return set()
    first, last = reads[0][0].start, reads[-1][0].end
    seen: dict[str, list[float]] = {}
    for shot, _, lines in reads:
        for _, _, text in lines:
            key = _norm_text(text)
            if len(key) >= 4:
                seen.setdefault(key, []).append(shot.start)
    return {
        key for key, times in seen.items()
        if len(times) >= 3 and (max(times) - min(times)) > 0.5 * (last - first)
    }


def _is_watermark(text: str, watermarks: set[str]) -> bool:
    key = _norm_text(text)
    if len(key) < 2:
        return False
    return any(key in w or difflib.SequenceMatcher(None, key, w).ratio() > 0.8 for w in watermarks)


def _clean_card_sequence(cards: list[TitleCard]) -> list[TitleCard]:
    """Drop nameless reads and stray numbers (intro/outro text), fix misread digits."""
    cards = [card for card in cards if card.name]
    # Some videos change the number and the name at slightly different moments, giving a
    # brief in-between read such as "2 <name of card 3>". Drop those transition reads.
    cards = [
        card for i, card in enumerate(cards)
        if card.end - card.start >= 0.6 or not any(
            0 <= j < len(cards) and _same_name(cards[j].name, card.name) for j in (i - 1, i + 1)
        )
    ]
    # A lower number directly after a card is almost always a misread digit (e.g. a 9 read
    # as 6) on that card's second dialogue, not a real step back.
    for prev, card in zip(cards, cards[1:]):
        if card.number < prev.number and card.start - prev.end <= 0.5:
            card.number = prev.number
    # Keep the longest run of non-decreasing numbers (1, 2, 3, 3, 4, ...); anything
    # outside it is stray text that happened to contain a number.
    n = len(cards)
    if n < 2:
        return cards
    best_len, prev_idx = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if cards[j].number <= cards[i].number and best_len[j] + 1 > best_len[i]:
                best_len[i], prev_idx[i] = best_len[j] + 1, j
    i = max(range(n), key=lambda k: best_len[k])
    keep = []
    while i != -1:
        keep.append(cards[i])
        i = prev_idx[i]
    return keep[::-1]


def _quietest_point(audio: AudioSegment, center_ms: int) -> int:
    """The quietest 10 ms within +/- BOUNDARY_SNAP_MS of center_ms, so cuts land between words."""
    lo = max(0, center_ms - BOUNDARY_SNAP_MS)
    hi = min(len(audio), center_ms + BOUNDARY_SNAP_MS)
    if hi - lo <= 10:
        return min(max(center_ms, 0), len(audio))
    best = min(range(lo, hi - 10, 10), key=lambda ms: audio[ms:ms + 10].rms)
    return best + 5


def _trim_silence(clip: AudioSegment) -> AudioSegment:
    lead = detect_leading_silence(clip, silence_threshold=SILENCE_THRESH_DB, chunk_size=10)
    if lead >= len(clip):
        return AudioSegment.empty()
    trail = detect_leading_silence(clip.reverse(), silence_threshold=SILENCE_THRESH_DB, chunk_size=10)
    return clip[max(0, lead - KEEP_SILENCE_MS):len(clip) - max(0, trail - KEEP_SILENCE_MS)]


def cut_clips_by_cards(audio: AudioSegment, cards: list[TitleCard]) -> list[tuple[str, AudioSegment]]:
    """One clip per card, from the card's appearance to its disappearance."""
    to_ms = lambda seconds: int(seconds * 1000)  # noqa: E731
    starts = [_quietest_point(audio, to_ms(cards[0].start))]
    ends = []
    for a, b in zip(cards, cards[1:]):
        if to_ms(b.start - a.end) < 2 * BOUNDARY_SNAP_MS:
            cut = _quietest_point(audio, to_ms((a.end + b.start) / 2))  # one shared cut point
            ends.append(cut)
            starts.append(cut)
        else:  # a long gap (intro, ad...) between cards: exclude it from both clips
            ends.append(_quietest_point(audio, to_ms(a.end)))
            starts.append(_quietest_point(audio, to_ms(b.start)))
    ends.append(_quietest_point(audio, to_ms(cards[-1].end)))

    clips = []
    for card, start, end in zip(cards, starts, ends):
        clip = _trim_silence(audio[start:end])
        stem = f"{card.number:02d}_{sanitize_filename(card.name) or 'dialogue'}"
        if len(clip) < 200:
            tqdm.write(f"  Card {card.number} has no audible dialogue; skipped.")
            continue
        clips.append((stem, clip))
    return clips


# =============================================================================
# STEP 2b: FALLBACK - SILENCE SPLIT
# =============================================================================
def split_on_pauses(audio: AudioSegment) -> list[AudioSegment]:
    tqdm.write(
        f"  Splitting {len(audio) / 1000:.1f}s of audio on silence "
        f"(min_silence_len={MIN_SILENCE_LEN_MS}ms, silence_thresh={SILENCE_THRESH_DB}dBFS)..."
    )
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=KEEP_SILENCE_MS,
        seek_step=SEEK_STEP_MS,
    )
    valid = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_LEN_MS]
    tqdm.write(
        f"  Found {len(chunks)} segments; kept {len(valid)}, "
        f"discarded {len(chunks) - len(valid)} shorter than {MIN_CHUNK_LEN_MS / 1000:.1f}s."
    )
    return valid


# =============================================================================
# STEP 3: MODELS (each loaded at most ONCE, and only when needed)
# =============================================================================
def load_whisper_model() -> WhisperModel:
    """Load the model once. Tries fp16 on GPU, then int8_float16 (less VRAM), then CPU."""
    attempts = [(DEVICE, COMPUTE_TYPE)]
    if DEVICE == "cuda":
        if ctranslate2.get_cuda_device_count() == 0:
            tqdm.write("WARNING: No CUDA device detected by CTranslate2. Check your NVIDIA driver.")
        if COMPUTE_TYPE != "int8_float16":
            attempts.append(("cuda", "int8_float16"))
        if ALLOW_CPU_FALLBACK:
            attempts.append(("cpu", "int8"))

    last_exc: Exception | None = None
    for device, compute_type in attempts:
        try:
            tqdm.write(f"  Loading Whisper '{WHISPER_MODEL_SIZE}' on {device} ({compute_type})...")
            model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
            tqdm.write(f"  Model ready on {device.upper()} ({compute_type}).")
            return model
        except (RuntimeError, ValueError) as exc:
            last_exc = exc
            if not _is_gpu_problem(exc):
                break  # not a GPU issue; retrying on another device won't help
            reason = "CUDA out of memory" if _is_oom(exc) else str(exc)
            tqdm.write(f"  Could not load on {device}/{compute_type}: {reason}")
        except OSError as exc:  # network / Hugging Face download / file errors
            raise PipelineError(f"Could not download or read the Whisper model: {exc}") from exc

    raise PipelineError(f"Failed to load Whisper model: {last_exc}") from last_exc


class DialogueClipper:
    """Holds the Whisper and OCR models so each is loaded once for the whole playlist."""

    def __init__(self) -> None:
        self._whisper: WhisperModel | None = None
        self._ocr: RapidOCR | None = None

    @property
    def whisper(self) -> WhisperModel:
        if self._whisper is None:
            self._whisper = load_whisper_model()
        return self._whisper

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            use_cuda = OCR_ON_GPU and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            onnxruntime.set_default_logger_severity(3)  # hide harmless provider-probe warnings
            if use_cuda and hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()  # CUDA/cuDNN DLLs from the nvidia-* pip wheels
            elif OCR_ON_GPU:
                tqdm.write("  WARNING: onnxruntime-gpu not available; OCR will run on the CPU (slower).")
            self._ocr = RapidOCR(params={
                "Global.log_level": "critical",
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
            })
        return self._ocr

    def close(self) -> None:
        self._whisper = None  # release VRAM
        self._ocr = None
        gc.collect()

    # ---------------- title cards ----------------
    def _read_number_above(self, frame: np.ndarray, bg: np.ndarray, text_top: int) -> int | None:
        """Fallback when full-frame OCR misses the big number (thick or low-contrast fonts):
        crop the area just above the name, keep pixels that differ from the background
        colour, and OCR that alone."""
        top, bottom = max(0, text_top - int(_SCAN_H * 0.35)), text_top - 2
        if bottom - top < 10:
            return None
        crop = frame[top:bottom, int(_SCAN_W * 0.25):int(_SCAN_W * 0.75)].astype(np.int16)
        ink = np.abs(crop - bg).sum(axis=-1) > 120
        if ink.mean() < 0.005:
            return None
        ys, xs = np.nonzero(ink)
        pad = 12
        bw = np.where(ink, 0, 255).astype(np.uint8)
        bw = bw[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
        bw = np.pad(bw, 20, constant_values=255)
        # use_* flags are passed on every call: RapidOCR remembers them between calls.
        result = self.ocr(np.stack([bw] * 3, axis=-1), use_det=False, use_cls=False, use_rec=True)
        match = _NUMBER_LINE.fullmatch((result.txts[0] if result.txts else "").strip())
        return int(match.group(1)) if match else None

    def _ocr_lines(self, frame: np.ndarray) -> list[tuple[float, float, str]]:
        """All text lines in the frame as (top, height, text), top to bottom."""
        result = self.ocr(frame, use_det=True, use_cls=True, use_rec=True)
        if not result.txts:
            return []
        return sorted(
            (float(box[:, 1].min()), float(box[:, 1].max() - box[:, 1].min()), text.strip())
            for box, text in zip(result.boxes, result.txts)
            if text.strip()
        )

    def _parse_card(self, frame: np.ndarray, bg: np.ndarray,
                    lines: list[tuple[float, float, str]]) -> tuple[int | None, str]:
        """Read (number, name) from a card's text lines; number is None if this is not a card."""
        number = None
        for i, (_, _, text) in enumerate(lines):
            match = _NUMBER_LINE.fullmatch(text)
            if match:
                number = int(match.group(1))
                lines = lines[i + 1:]  # the name is below the number
                break

        # Drop junk: lines without real words, and small watermark/channel-name text.
        lines = [line for line in lines if len(re.findall(r"[^\W\d_]", line[2])) >= 2]
        if lines:
            tallest = max(height for _, height, _ in lines)
            lines = [line for line in lines if line[1] >= 0.5 * tallest]
        if number is None and lines:
            number = self._read_number_above(frame, bg, int(lines[0][0]))
        return number, " ".join(text for _, _, text in lines)

    def detect_title_cards(self, video_path: Path, duration: float | None) -> list[TitleCard]:
        # Pass 1: OCR one frame of every plain-background shot.
        reads = []
        for shot in _split_into_shots(video_path, duration):
            bg, fraction = _card_background(shot.frame)
            if fraction >= CARD_MIN_BG_FRACTION:
                lines = self._ocr_lines(shot.frame)
                if lines:
                    reads.append((shot, bg, lines))
        watermarks = _find_watermarks(reads)

        # Pass 2: turn the reads into cards.
        cards: list[TitleCard] = []
        for shot, bg, lines in reads:
            lines = [line for line in lines if not _is_watermark(line[2], watermarks)]
            number, name = self._parse_card(shot.frame, bg, lines)
            if number is None:
                continue  # intro text, "Download link in description", outro, transitions
            last = cards[-1] if cards else None
            if last and last.number == number:
                gap = shot.start - last.end
                if (name and last.name and _same_name(name, last.name) and gap <= 2.0) or \
                        ((not name or not last.name) and gap <= 0.5):
                    last.end = shot.end  # same card, e.g. after a small animation
                    last.name = last.name or name
                    continue
            # A new number, or the same number with a different name (a second dialogue).
            cards.append(TitleCard(number, name, shot.start, shot.end))
        return _clean_card_sequence(cards)

    # ---------------- Whisper fallback ----------------
    def _transcribe_first_words(self, chunk: AudioSegment) -> str:
        mono = chunk.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        samples = np.frombuffer(mono.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(
            samples,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=False,                 # chunks are already split on silence
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        words: list[str] = []
        for segment in segments:  # lazy generator: stop decoding once we have enough words
            words.extend(segment.text.split())
            if len(words) >= MAX_WORDS_IN_NAME:
                break
        text = " ".join(words[:MAX_WORDS_IN_NAME])
        return devanagari_to_malayalam(text) if LANGUAGE == "ml" else text

    def name_chunks_with_whisper(self, chunks: list[AudioSegment]) -> list[tuple[str, AudioSegment]]:
        self.whisper  # load before the progress bar starts  # noqa: B018
        named = []
        for idx, chunk in enumerate(tqdm(chunks, desc="Transcribing", unit="clip",
                                         dynamic_ncols=True, leave=False), start=1):
            text = ""
            try:
                text = self._transcribe_first_words(chunk)
            except RuntimeError as exc:
                reason = "CUDA out of memory" if _is_oom(exc) else str(exc)
                tqdm.write(f"  [{idx}] Transcription failed ({reason}); using fallback name.")
            except Exception as exc:  # never let one bad clip kill the whole run
                tqdm.write(f"  [{idx}] Unexpected transcription error: {exc}; using fallback name.")
            named.append((sanitize_filename(text) or f"dialogue_{idx:03d}", chunk))
        return named

    # ---------------- one video, end to end ----------------
    def process_video(self, job: VideoJob) -> VideoResult:
        result = VideoResult(job)
        final_dir = OUTPUT_DIR / job.folder_name
        if SKIP_COMPLETED_VIDEOS and final_dir.is_dir():
            result.status = "skipped (folder already exists)"
            return result

        # Build into a ".partial" folder and rename at the end, so an interrupted run is
        # never mistaken for a finished one.
        work_dir = final_dir.with_name(final_dir.name + ".partial")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        temp_files: list[Path] = []
        duration = None
        if job.local_path:
            audio_path = job.local_path
        else:
            audio_path = download_audio(job)
            temp_files.append(audio_path)
        audio = load_audio(audio_path)
        duration = len(audio) / 1000

        clips: list[tuple[str, AudioSegment]] = []
        if USE_TITLE_CARDS and not job.local_path:
            try:
                video_path = download_card_video(job)
                temp_files.append(video_path)
                cards = self.detect_title_cards(video_path, duration)
            except PipelineError as exc:
                tqdm.write(f"  Title-card detection unavailable ({exc}); falling back to Whisper.")
                cards = []
            if cards:
                tqdm.write(f"  Found {len(cards)} title cards: "
                           + ", ".join(f"{c.number} {c.name!r}" for c in cards))
                clips = cut_clips_by_cards(audio, cards)
                result.status = f"{len(cards)} title cards"
            else:
                tqdm.write("  No title cards found; using silence split + Whisper naming.")

        if not clips:
            chunks = split_on_pauses(audio)
            del audio
            if not chunks:
                raise PipelineError(
                    "No usable segments found. Try a higher SILENCE_THRESH_DB (e.g. -35) "
                    "or a lower MIN_CHUNK_LEN_MS."
                )
            clips = self.name_chunks_with_whisper(chunks)
            result.status = "silence split + Whisper"

        result.created, result.failed = export_clips(clips, work_dir)
        if result.failed:
            result.ok = False
            result.status += f", {result.failed} export(s) failed; kept {work_dir.name}"
            return result

        work_dir.rename(final_dir)
        if DELETE_RAW_AFTER_SUCCESS:
            for path in temp_files:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    tqdm.write(f"  WARNING: Could not delete {path.name}: {exc}")
        return result


# =============================================================================
# STEP 4: EXPORT
# =============================================================================
def export_chunk(chunk: AudioSegment, path: Path) -> None:
    if EXPORT_FORMAT == "mp3":
        handle = chunk.export(path, format="mp3", bitrate=MP3_BITRATE)
    else:
        handle = chunk.export(path, format="wav")
    handle.close()  # pydub returns an open handle; close it so Windows releases the file


def export_clips(clips: list[tuple[str, AudioSegment]], out_dir: Path) -> tuple[int, int]:
    """Returns (clips_created, clips_failed)."""
    created = failed = 0
    ext = EXPORT_FORMAT.lower()
    for stem, clip in tqdm(clips, desc="Saving", unit="clip", dynamic_ncols=True, leave=False):
        out_path = unique_path(out_dir, stem, ext)
        try:
            export_chunk(clip, out_path)
            created += 1
        except (OSError, CouldntEncodeError) as exc:
            failed += 1
            tqdm.write(f"  Failed to export '{out_path.name}': {exc}")
    return created, failed


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YouTube audio into named dialogue clips.")
    parser.add_argument("url", nargs="?", default=None,
                        help="YouTube video or playlist URL (overrides YOUTUBE_URL).")
    parser.add_argument("--file", type=Path, default=None,
                        help="Process an existing local audio file instead of downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()

    try:
        check_ffmpeg()
        if args.file:
            path = args.file.resolve()
            if not path.exists():
                raise PipelineError(f"File not found: {path}")
            jobs = [VideoJob(1, path.stem, path.stem, "", local_path=path)]
        else:
            url = args.url or YOUTUBE_URL
            if not url:
                raise PipelineError("Set YOUTUBE_URL at the top of main.py or pass a URL argument.")
            print("Reading URL...")
            jobs = list_videos(url)
    except PipelineError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{len(jobs)} video(s) to process. Output: {OUTPUT_DIR}\n")
    clipper = DialogueClipper()
    results: list[VideoResult] = []
    try:
        for n, job in enumerate(jobs, start=1):
            print(f"[{n}/{len(jobs)}] {job.folder_name}  {job.title}")
            try:
                result = clipper.process_video(job)
            except PipelineError as exc:
                result = VideoResult(job, status=f"ERROR: {exc}", ok=False)
            except Exception as exc:  # one broken video must not stop the whole playlist
                result = VideoResult(job, status=f"ERROR (unexpected): {exc!r}", ok=False)
            results.append(result)
            print(f"  -> {result.created} clips ({result.status})\n")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Re-run the same command to resume.", file=sys.stderr)
        return 130
    finally:
        clipper.close()

    total = sum(r.created for r in results)
    failures = [r for r in results if not r.ok]
    print("=" * 70)
    for r in results:
        print(f"  {'OK ' if r.ok else 'ERR'}  {r.job.folder_name:<22} {r.created:>3} clips  {r.status}")
    print("-" * 70)
    print(f"Successfully created {total} audio clips from {len(results) - len(failures)} video(s)!")
    if failures:
        print(f"{len(failures)} video(s) had problems; re-run to retry them.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 70)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
