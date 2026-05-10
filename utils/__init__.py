"""
Utils package initialisation.

Bootstraps an ffmpeg binary for pydub so audio/video processing works
reliably on every environment:

  1. If the OS already has ``ffmpeg`` on PATH (e.g. Streamlit Community
     Cloud after ``packages.txt`` installs it, or a local ``brew install
     ffmpeg``), pydub picks it up automatically.
  2. Otherwise we fall back to the static binary shipped by the
     ``imageio-ffmpeg`` Python wheel and point pydub at it explicitly.
     This means ``pip install -r requirements.txt`` alone is enough to
     get audio extraction working — no system package required.

The bootstrap is best-effort and never raises: if neither path is
available, downstream callers degrade gracefully (e.g. video ingestion
proceeds with vision only, no audio context).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings


def _ensure_ffmpeg_named_alias(ffmpeg_exe: str) -> str:
    """
    Create (once) a stable ``ffmpeg``-named symlink (or copy on Windows /
    restricted FS) pointing at ``ffmpeg_exe`` and return the directory
    containing it. This makes pydub's own ``shutil.which("ffmpeg")``
    lookup succeed even when the underlying binary has a versioned name
    like ``ffmpeg-macos-aarch64-v7.1`` (as shipped by ``imageio-ffmpeg``).
    """
    link_dir = os.path.join(tempfile.gettempdir(), "ragnarok_ffmpeg_bin")
    os.makedirs(link_dir, exist_ok=True)

    link_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    link_path = os.path.join(link_dir, link_name)

    if not os.path.exists(link_path):
        try:
            os.symlink(ffmpeg_exe, link_path)
        except (OSError, NotImplementedError, AttributeError):
            # Symlinks unsupported (e.g. Windows without admin) — copy instead.
            try:
                shutil.copy2(ffmpeg_exe, link_path)
                os.chmod(link_path, 0o755)
            except Exception:
                return os.path.dirname(ffmpeg_exe)

    return link_dir


def _bootstrap_ffmpeg() -> None:
    # 1. System ffmpeg already available → nothing to do. This is the path
    #    taken on Streamlit Community Cloud (after ``packages.txt`` installs
    #    ffmpeg) and on any machine with ``brew install ffmpeg``.
    if shutil.which("ffmpeg"):
        return

    # 2. Fall back to the bundled imageio-ffmpeg binary.
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return

    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        return

    # Expose the binary on PATH under the canonical name ``ffmpeg`` so
    # pydub's module-level discovery succeeds and no RuntimeWarning fires.
    ffmpeg_dir = _ensure_ffmpeg_named_alias(ffmpeg_exe)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    # Defensive: silence pydub's discovery warning in case this bootstrap
    # ever runs *after* pydub has already been imported elsewhere.
    warnings.filterwarnings(
        "ignore",
        message=r"Couldn't find ffmpeg or avconv.*",
        category=RuntimeWarning,
        module=r"pydub\.utils",
    )

    # Also configure pydub directly in case PATH lookup is bypassed.
    try:
        from pydub import AudioSegment  # type: ignore
        AudioSegment.converter = ffmpeg_exe
        AudioSegment.ffmpeg = ffmpeg_exe
        # imageio-ffmpeg doesn't ship ffprobe; if the system also lacks
        # it, pydub callers should pass an explicit ``format=`` hint so
        # ffprobe is never invoked.
        if not shutil.which("ffprobe"):
            AudioSegment.ffprobe = ffmpeg_exe  # harmless placeholder
    except Exception:
        pass


_bootstrap_ffmpeg()
