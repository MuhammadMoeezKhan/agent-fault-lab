"""Audit target for `io.github.korovin-aa97/talkthrough-mcp`, frame position 470.

Written in Python rather than JSON for one reason: the fixture needs a real
media file to ingest, and a JSON spec's `setup` block can only write text. The
seed here is a one second mono WAV synthesised in process, so nothing is
downloaded and the trial stays offline under rule 9.

Everything else is the ordinary contained third-party shape: `contained` stays
at its default True, and the image is the one carrying `ffmpeg` from the
distribution at build time plus the package installed with `uv tool install`,
since this is a Python distribution rather than an npm package.

`process_media` is measured twice, once with its arguments left at their
defaults and once with `force: true`, so the same tool is covered with and
without the guard its own boolean controls. The run's outcome and what it says
about the server are in the private research notes, not here.
"""

from __future__ import annotations

import math
import os
import struct
import wave

from agentfaultlab.audit import Fixture, ServerTarget
from agentfaultlab.observers import FilesystemObserver

#: One second, 16 kHz, mono, a 440 Hz tone. Whisper takes 16 kHz mono
#: directly, so no resampling step is needed, and a tone rather than silence
#: gives the pipeline something non-degenerate to look at.
RATE = 16000
SECONDS = 1.0


def _write_wav(path: str) -> None:
    frames = bytearray()
    for index in range(int(RATE * SECONDS)):
        value = int(12000 * math.sin(2 * math.pi * 440 * index / RATE))
        frames += struct.pack("<h", value)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))


def _setup(root: str) -> None:
    os.makedirs(os.path.join(root, "home"), exist_ok=True)
    _write_wav(os.path.join(root, "clip.wav"))


def _observer(root: str) -> FilesystemObserver:
    """The whole trial root, with no drops declared yet.

    The store's location is not assumed: HOME is moved into the trial root so
    nothing reaches the operator's home, and the first runs observe home as
    well, so that where the effect actually lands is measured rather than
    guessed. Any projection added later has to come from diffing two
    identical clean runs and naming what moved, per preregistration 6.3.
    """
    return FilesystemObserver(root, ignore=(".DS_Store",))


TARGET = ServerTarget(
    name="community-talkthrough",
    command=["uvx", "talkthrough-mcp"],
    container_command=["talkthrough-mcp"],
    version="talkthrough 0.4.4 (uv installed at image build time)",
    observer=_observer,
    setup=_setup,
    env={"HOME": "{root}/home", "NO_COLOR": "1"},
    timeout_s=300.0,
    image="agent-fault-lab:block5-ffmpeg-py",
)

FIXTURES = [
    Fixture(
        tool="process_media",
        id="process_media_content_addressed",
        arguments={"path": "{root}/clip.wav"},
        probes=(("list_jobs", {}),),
        note=(
            "The documented happy path: ingest a local file with every "
            "optional argument left at its default. The store is "
            "content-addressed, so a second application of the same call "
            "meets a job that already exists for those bytes."
        ),
    ),
    Fixture(
        tool="process_media",
        id="process_media_force_true",
        arguments={"path": "{root}/clip.wav", "force": True},
        probes=(("list_jobs", {}),),
        note=(
            "The same call with the content-addressed guard turned off by a "
            "caller-passed boolean. One hint on the tool has to describe both "
            "this and the fixture above."
        ),
    ),
]
