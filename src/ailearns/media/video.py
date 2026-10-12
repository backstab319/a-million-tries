"""Write pygame frames straight into an mp4 through an ffmpeg pipe."""

import subprocess
from pathlib import Path

import pygame


class VideoWriter:
    def __init__(self, path: str | Path, size: tuple[int, int], fps: int = 60, crf: int = 18,
                 preset: str = "medium"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.frames = 0
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, surface: pygame.Surface, repeat: int = 1) -> None:
        assert surface.get_size() == self.size, (surface.get_size(), self.size)
        data = pygame.image.tobytes(surface, "RGB")
        for _ in range(repeat):
            self.proc.stdin.write(data)
            self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
