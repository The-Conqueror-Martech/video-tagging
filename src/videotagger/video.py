"""Video processing for frame extraction."""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from videotagger.exceptions import VideoProcessingError

logger = logging.getLogger(__name__)


def extract_frames(video_path: str | Path, num_frames: int = 8) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract (default: 8).

    Returns:
        List of frames as numpy arrays (BGR format).

    Raises:
        VideoProcessingError: If video cannot be opened or read.
    """
    video_path = Path(video_path)
    logger.info(f"Extracting {num_frames} frames from: {video_path}")

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise VideoProcessingError(f"Video file not found: {video_path}", str(video_path))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        raise VideoProcessingError(f"Could not open video: {video_path}", str(video_path))

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}, {duration:.1f}s"
        )

        if total_frames < 1:
            logger.error(f"Video has no frames: {video_path}")
            raise VideoProcessingError(f"Video has no frames: {video_path}", str(video_path))

        # Calculate frame indices to extract (evenly spaced)
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue  # Skip frames that can't be read

            frames.append(frame)

        if not frames:
            logger.error(f"Could not extract any frames from video: {video_path}")
            raise VideoProcessingError(
                f"Could not extract any frames from video: {video_path}",
                str(video_path),
            )

        logger.info(f"Successfully extracted {len(frames)} frames")
        return frames

    finally:
        cap.release()


def frame_to_base64(frame: np.ndarray, format: str = "jpg", max_size: int = 768) -> str:
    """Convert a frame to base64-encoded string.

    Args:
        frame: Frame as numpy array (BGR format from OpenCV).
        format: Image format for encoding ('jpg' or 'png').
        max_size: Maximum dimension (width or height) in pixels. Larger images are downsampled.

    Returns:
        Base64-encoded string of the image.

    Raises:
        VideoProcessingError: If encoding fails.
    """
    # Downsample if needed to reduce token usage
    height, width = frame.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.debug(f"Downsampled frame from {width}x{height} to {new_width}x{new_height}")

    if format.lower() == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    else:
        ext = ".png"
        params = []

    success, buffer = cv2.imencode(ext, frame, params)

    if not success:
        raise VideoProcessingError("Failed to encode frame to image")

    return base64.b64encode(buffer).decode("utf-8")


def extract_frames_as_base64(
    video_path: str | Path,
    num_frames: int = 8,
    max_size: int = 768,
) -> list[str]:
    """Extract frames from video and return as base64-encoded strings.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.
        max_size: Maximum dimension for frames (default 768px).

    Returns:
        List of base64-encoded JPEG images.

    Raises:
        VideoProcessingError: If extraction or encoding fails.
    """
    frames = extract_frames(video_path, num_frames)
    return [frame_to_base64(frame, max_size=max_size) for frame in frames]


@dataclass
class WeightedFrames:
    """Container for weighted frame extraction results."""

    hook_frames: list[np.ndarray]
    context_frames: list[np.ndarray]
    fps: float
    duration: float

    @property
    def total_frames(self) -> int:
        return len(self.hook_frames) + len(self.context_frames)


@dataclass
class WeightedFramesBase64:
    """Container for weighted frames as base64 strings."""

    hook_frames: list[str]
    context_frames: list[str]

    @property
    def all_frames(self) -> list[str]:
        """Return all frames with hook frames first."""
        return self.hook_frames + self.context_frames


def extract_weighted_frames(
    video_path: str | Path,
    hook_interval: float = 0.5,
    hook_duration: float = 1.5,
    context_interval: float = 2.0,
) -> WeightedFrames:
    """Extract frames with dense sampling in first seconds (hook) and sparse sampling after.

    This implements the "Visual Hook" strategy where the first few seconds of video
    are sampled densely (every hook_interval seconds) while the remainder is sampled
    sparsely (every context_interval seconds).

    Args:
        video_path: Path to the video file.
        hook_interval: Interval between frames in hook period (default: 0.5s).
        hook_duration: Duration of the hook period (default: 1.5s, yields 3 frames at 0.0, 0.5, 1.0).
        context_interval: Interval between frames after hook period (default: 2.0s).

    Returns:
        WeightedFrames with hook_frames and context_frames separated.

    Raises:
        VideoProcessingError: If video cannot be opened or read.
    """
    video_path = Path(video_path)
    logger.info(f"Extracting weighted frames from: {video_path}")

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise VideoProcessingError(f"Video file not found: {video_path}", str(video_path))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        raise VideoProcessingError(f"Could not open video: {video_path}", str(video_path))

    try:
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frame_count / fps if fps > 0 else 0

        logger.info(f"Video info: {total_frame_count} frames, {fps:.1f} fps, {duration:.1f}s")

        if total_frame_count < 1:
            logger.error(f"Video has no frames: {video_path}")
            raise VideoProcessingError(f"Video has no frames: {video_path}", str(video_path))

        # Calculate hook frame timestamps (0.0, 0.5, 1.0, etc. up to hook_duration)
        hook_timestamps: list[float] = []
        t = 0.0
        while t < min(hook_duration, duration):
            hook_timestamps.append(t)
            t += hook_interval

        # Calculate context frame timestamps (starting after hook_duration)
        context_timestamps: list[float] = []
        t = hook_duration + context_interval
        while t < duration:
            context_timestamps.append(t)
            t += context_interval

        # For very short videos (< hook_duration), treat entire video as hook
        if duration <= hook_duration:
            # Sample densely for entire video
            hook_timestamps = []
            t = 0.0
            while t < duration:
                hook_timestamps.append(t)
                t += hook_interval
            # Ensure we get at least the first frame
            if not hook_timestamps:
                hook_timestamps = [0.0]
            context_timestamps = []

        logger.debug(f"Hook timestamps: {hook_timestamps}")
        logger.debug(f"Context timestamps: {context_timestamps}")

        def extract_at_timestamps(timestamps: list[float]) -> list[np.ndarray]:
            frames = []
            for ts in timestamps:
                frame_idx = int(ts * fps)
                frame_idx = min(frame_idx, total_frame_count - 1)  # Clamp to valid range
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            return frames

        hook_frames = extract_at_timestamps(hook_timestamps)
        context_frames = extract_at_timestamps(context_timestamps)

        if not hook_frames and not context_frames:
            logger.error(f"Could not extract any frames from video: {video_path}")
            raise VideoProcessingError(
                f"Could not extract any frames from video: {video_path}",
                str(video_path),
            )

        logger.info(
            f"Extracted {len(hook_frames)} hook frames + {len(context_frames)} context frames"
        )

        return WeightedFrames(
            hook_frames=hook_frames,
            context_frames=context_frames,
            fps=fps,
            duration=duration,
        )

    finally:
        cap.release()


def extract_weighted_frames_as_base64(
    video_path: str | Path,
    hook_interval: float = 0.5,
    hook_duration: float = 1.5,
    context_interval: float = 2.0,
    max_size: int = 768,
) -> WeightedFramesBase64:
    """Extract weighted frames and return as base64-encoded strings.

    Args:
        video_path: Path to the video file.
        hook_interval: Interval between frames in hook period (default: 0.5s).
        hook_duration: Duration of the hook period (default: 1.5s).
        context_interval: Interval between frames after hook period (default: 2.0s).
        max_size: Maximum dimension for frames (default 768px).

    Returns:
        WeightedFramesBase64 with hook_frames and context_frames as base64 strings.

    Raises:
        VideoProcessingError: If extraction or encoding fails.
    """
    weighted = extract_weighted_frames(
        video_path,
        hook_interval=hook_interval,
        hook_duration=hook_duration,
        context_interval=context_interval,
    )

    return WeightedFramesBase64(
        hook_frames=[frame_to_base64(f, max_size=max_size) for f in weighted.hook_frames],
        context_frames=[frame_to_base64(f, max_size=max_size) for f in weighted.context_frames],
    )
