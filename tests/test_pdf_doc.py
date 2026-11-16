"""PDF documents, compared with declared trailer keys given up.

A renderer stamps each PDF with a creation date and a file identifier, so two
renders of the same input differ in bytes and every repeat would read as a
change. These tests pin the contract: only the values of declared keys are
given up, whether written directly or through an indirect reference, and any
other difference, including one in page content, still shows.
"""

from agentfaultlab.observers import PdfDocObserver, equal
from agentfaultlab.spec import _observer_factory


def pdf(date: bytes, ident: bytes, body: bytes = b"BT (hello) Tj ET") -> bytes:
    # The shape a headless Chromium render has: the info dictionary refers to
    # the date indirectly, and the trailer carries /ID directly.
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"3 0 obj\n<< /Length 16 >>\nstream\n" + body + b"\nendstream\nendobj\n"
        b"16 0 obj\n<< /Producer (Skia/PDF) /CreationDate 17 0 R /ModDate 17 0 R >>\nendobj\n"
        b"17 0 obj\n(D:" + date + b"Z)\nendobj\n"
        b"trailer\n<< /Root 1 0 R /Info 16 0 R\n/ID [<" + ident + b"> <" + ident + b">] >>\n%%EOF\n"
    )


DROPS = ("CreationDate", "ModDate", "ID")


def write(directory, name, data):
    directory.mkdir(exist_ok=True)
    (directory / name).write_bytes(data)


def test_two_renders_of_one_input_are_equal_once_declared(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "note.pdf", pdf(b"20261116162007", b"2c64080dfa309129bd837dd3bca5acc4"))
    write(b, "note.pdf", pdf(b"20261116162116", b"a895d3cb69ca5264c668df251561c3ed"))
    assert equal(
        PdfDocObserver(str(a), drop_keys=DROPS).snapshot(),
        PdfDocObserver(str(b), drop_keys=DROPS).snapshot(),
    )


def test_without_the_declaration_the_stamps_are_state(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "note.pdf", pdf(b"20261116162007", b"2c64"))
    write(b, "note.pdf", pdf(b"20261116162116", b"a895"))
    assert not equal(PdfDocObserver(str(a)).snapshot(), PdfDocObserver(str(b)).snapshot())


def test_dropping_only_the_date_leaves_the_identifier_compared(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "note.pdf", pdf(b"20261116162007", b"2c64"))
    write(b, "note.pdf", pdf(b"20261116162116", b"a895"))
    assert not equal(
        PdfDocObserver(str(a), drop_keys=("CreationDate", "ModDate")).snapshot(),
        PdfDocObserver(str(b), drop_keys=("CreationDate", "ModDate")).snapshot(),
    )


def test_a_change_in_page_content_still_shows(tmp_path):
    """The guard against an observer that launders away a real difference."""
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "note.pdf", pdf(b"20261116162007", b"2c64", b"BT (hello) Tj ET"))
    write(b, "note.pdf", pdf(b"20261116162007", b"2c64", b"BT (hellp) Tj ET"))
    assert not equal(
        PdfDocObserver(str(a), drop_keys=DROPS).snapshot(),
        PdfDocObserver(str(b), drop_keys=DROPS).snapshot(),
    )


def test_a_second_file_under_a_new_name_shows_in_tally_mode(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "note-2026-11-16T16-20-07.pdf", pdf(b"20261116162007", b"aa"))
    write(b, "note-2026-11-16T16-21-16.pdf", pdf(b"20261116162116", b"bb"))
    one_a = PdfDocObserver(str(a), drop_keys=DROPS, tally=True).snapshot()
    one_b = PdfDocObserver(str(b), drop_keys=DROPS, tally=True).snapshot()
    assert equal(one_a, one_b), "names are not state in tally mode"
    write(b, "note-2026-11-16T16-21-30.pdf", pdf(b"20261116162130", b"cc"))
    two_b = PdfDocObserver(str(b), drop_keys=DROPS, tally=True).snapshot()
    assert two_b["count"] == 2
    assert not equal(one_a, two_b)


def test_names_are_state_outside_tally_mode(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "one.pdf", pdf(b"20261116162007", b"aa"))
    write(b, "two.pdf", pdf(b"20261116162007", b"aa"))
    assert not equal(
        PdfDocObserver(str(a), drop_keys=DROPS).snapshot(),
        PdfDocObserver(str(b), drop_keys=DROPS).snapshot(),
    )


def test_the_spec_kind_builds_it(tmp_path):
    factory = _observer_factory(
        {"kind": "pdf-doc", "path": "out", "drop_keys": list(DROPS), "tally": True}
    )
    observer = factory(str(tmp_path))
    assert isinstance(observer, PdfDocObserver)
    assert observer.tally and observer.drop_keys == tuple(sorted(DROPS))
    assert observer.snapshot()["count"] == 0
