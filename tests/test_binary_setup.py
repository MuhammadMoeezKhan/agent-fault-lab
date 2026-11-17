"""Binary seed files in a spec's setup, written from base64."""

import base64

import pytest

from agentfaultlab.spec import SpecError
from agentfaultlab.spec import load as load_spec

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _spec(tmp_path, binary: str) -> str:
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"server": {"name": "x", "command": ["true"], "observer": {"kind": "filesystem"},'
        ' "setup": {"binary_files": {"img/a.png": "' + binary + '"}}},'
        ' "fixtures": [{"tool": "t", "arguments": {}}]}'
    )
    return str(spec)


def test_a_binary_seed_is_written_byte_for_byte(tmp_path):
    target, _ = load_spec(_spec(tmp_path, base64.b64encode(PNG).decode()))
    root = tmp_path / "root"
    root.mkdir()
    target.setup(str(root))
    assert (root / "img" / "a.png").read_bytes() == PNG


def test_a_seed_that_is_not_base64_is_refused(tmp_path):
    target, _ = load_spec(_spec(tmp_path, "not base64!"))
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(SpecError, match="not base64"):
        target.setup(str(root))


def test_the_test_file_writes_no_em_dashes():
    with open(__file__, encoding="utf-8") as handle:
        assert chr(0x2014) not in handle.read()
