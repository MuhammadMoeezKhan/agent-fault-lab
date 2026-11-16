"""Audit target for `io.github.benbalter/word-to-markdown`, frame position 572.

Written in Python rather than JSON because the fixture needs a real .docx to
convert, and a JSON spec's `setup` block can only write text. The seed is a
minimal WordprocessingML package (content types, one relationship, one
document part with a heading and a paragraph) built in process with fixed
entry timestamps, so nothing is downloaded and the trial stays offline under
rule 9.

The server is started with the registry's `mcp` argument. Its one tool takes
an `output_path` and an `overwrite` boolean that is off by default, so the
same tool is measured with and without the guard its own boolean controls.
A third fixture omits `output_path`, so the Markdown is returned rather than
written. The run's outcome is in the private research notes, not here.
"""

from __future__ import annotations

import os
import zipfile

from agentfaultlab.audit import Fixture, ServerTarget
from agentfaultlab.observers import FilesystemObserver

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/></Relationships>'
)
_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Audit memo</w:t></w:r></w:p>'
    "<w:p><w:r><w:t>One paragraph of body text.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def _write_docx(path: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in (
            ("[Content_Types].xml", _CONTENT_TYPES),
            ("_rels/.rels", _RELS),
            ("word/document.xml", _DOCUMENT),
        ):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), body)


def _setup(root: str) -> None:
    os.makedirs(os.path.join(root, "home"), exist_ok=True)
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    _write_docx(os.path.join(root, "docs", "memo.docx"))


def _observer(root: str) -> FilesystemObserver:
    """The trial root, with HOME and the macOS folder litter left out."""
    return FilesystemObserver(root, ignore=("home", "home/*", ".DS_Store"))


TARGET = ServerTarget(
    name="community-word-to-markdown",
    command=["npx", "-y", "word-to-markdown", "mcp"],
    container_command=["word-to-markdown", "mcp"],
    version="word-to-markdown (npm latest at image build time)",
    observer=_observer,
    setup=_setup,
    env={"HOME": "{root}/home", "NO_COLOR": "1"},
    timeout_s=120.0,
    image="agent-fault-lab:block5-n7",
)

FIXTURES = [
    Fixture(
        tool="convert_docx_to_markdown",
        id="convert_to_new_file_guarded",
        arguments={"path": "{root}/docs/memo.docx", "output_path": "{root}/out/memo.md"},
        note="Writes the Markdown to a new file with overwrite left at its default, off.",
    ),
    Fixture(
        tool="convert_docx_to_markdown",
        id="convert_to_new_file_overwrite_true",
        arguments={
            "path": "{root}/docs/memo.docx",
            "output_path": "{root}/out/memo.md",
            "overwrite": True,
        },
        note="The same call with the guard turned off by the caller-passed boolean.",
    ),
    Fixture(
        tool="convert_docx_to_markdown",
        id="convert_returned_inline",
        arguments={"path": "{root}/docs/memo.docx"},
        note="output_path omitted, so the Markdown is returned in the reply.",
    ),
]
