"""scripts/export_public.py: what stays out of the public copy, and what the scan flags."""

import codecs
import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_public.py"
spec = importlib.util.spec_from_file_location("export_public", SCRIPT)
export_public = importlib.util.module_from_spec(spec)
sys.modules["export_public"] = export_public  # its dataclass looks its module up while the class is made
spec.loader.exec_module(export_public)


def test_excluded_files_and_folders_say_why_and_the_rest_go_in():
    assert export_public.excluded("Launch-Presentation.pdf")
    assert export_public.excluded("media/ladder-test/ladder.html")
    assert export_public.excluded("media/ladder-testing/notes.md") is None
    assert export_public.excluded("src/handoff/bt.py") is None


def test_the_scan_flags_home_paths_chat_links_keys_and_local_words(tmp_path):
    local = tmp_path / "words.txt"
    local.write_text("# comment\n(?i)\\bjane\\b\n\n", encoding="utf-8")
    patterns = export_public.load_patterns(local)
    text = "\n".join(
        [
            "weights in C:\\Users\\someone\\models",
            "see https://claude.ai/code/session_abc",
            "token ghp_" + "a" * 24,
            "written by Jane",
            "Co-Authored-By: Claude <noreply@anthropic.com>",
            "plain line about the robot",
        ]
    )
    kinds = {h.line: h.kind for h in export_public.scan_text("x.md", text, patterns)}
    assert kinds == {1: "home path", 2: "claude.ai link", 3: "api key", 4: "private word"}


def kinds(line):
    return [h.kind for h in export_public.scan_text("x.py", line, export_public.load_patterns(Path("missing")))]


@pytest.mark.parametrize(
    "line",
    [
        r"weights in C:\Users\someone",  # one backslash: a raw string, a doc
        r'"C:\\Users\\someone\\models"',  # doubled: a Python or JSON string
        "weights in C:/Users/someone",
        r"\\wsl.localhost\Ubuntu\home\someone\dev",
        "/home/someone/dev",
        "/mnt/c/Users/someone/dev",
        "\x1b[0;32mC:\\Users\\someone\\x.py",  # a coloured traceback in a notebook output
        r'"msg\nC:\\Users\\someone"',  # after an escaped newline in a JSON string
        "C://Users/someone",  # doubled slash after the colon is still a path, not a URL scheme
        "C://Users//",  # the same with no name after it, which only the drive form catches
        "://hostname.local/Users/someone/junk.usd",  # a drive-less /Users/name, as in MH3.usd
        "see ~/dev/project/NOW.md",
        r"in ~\dev\project",
    ],
)
def test_the_home_path_pattern_matches_every_spelling_of_the_separator(line):
    assert kinds(line) == ["home path"]


@pytest.mark.parametrize(
    "line",
    [
        r"D:\Models\hf",
        r'"D:\\Models\\decision-servers"',
        "D:/Models/hf",
        "d:\\toolkits",
        "/mnt/d/Models/hf",
        r"\\wsl.localhost\Ubuntu\mnt\d\Ollama",
        r"File \x1b[1;32mD:\\Models\\hf",
    ],
)
def test_the_local_drive_pattern_matches_every_spelling_of_the_separator(line):
    assert kinds(line) == ["local drive path"]


@pytest.mark.parametrize(
    "line",
    ["see https://ollama.com", "http://models.example", "file:///toolkits", "ws://ollama.local", "gs://models/bert"],
)
def test_a_url_is_not_a_drive_path(line):
    assert kinds(line) == []


def test_text_is_found_by_its_bytes_not_its_suffix():
    """A notebook (.ipynb) or any other new text type is scanned; a binary file (a NUL byte) is not; a text file that is
    not UTF-8 is still scanned, not skipped whole."""
    assert "Users" in export_public.text_of(b'{"cells": [{"outputs": ["C:\\\\Users\\\\someone"]}]}')
    assert export_public.text_of(b"") == ""
    assert export_public.text_of(b"\x89PNG\r\n\x1a\n\x00\x00") is None
    assert "C:\\Users" in export_public.text_of(b"caf\xe9 at C:\\Users\\someone")  # cp1252, from PowerShell 5.1
    assert "C:\\Users" in export_public.text_of("C:\\Users\\someone".encode("utf-16"))  # with its byte-order mark


@pytest.mark.parametrize(
    "data",
    ["C:\\Users\\someone".encode("utf-32"), codecs.BOM_UTF32_BE + "C:\\Users\\someone".encode("utf-32-be")],
    ids=["native", "big-endian"],
)
def test_utf32_with_a_byte_order_mark_is_scanned_not_misread_as_utf16(data):
    """UTF-32-LE's mark starts with UTF-16-LE's; read as UTF-16 every character gets a NUL beside it and nothing matches."""
    assert export_public.text_of(data) == "C:\\Users\\someone"


def test_the_excerpt_holds_a_match_far_along_a_long_line():
    line = "x" * 300 + " weights in D:\\Models\\hf-win " + "y" * 300
    (hit,) = export_public.scan_text("x.md", line, export_public.load_patterns(Path("missing")))
    assert hit.kind == "local drive path"
    assert "D:\\Models\\hf-win" in hit.text
    assert hit.text.startswith("...") and hit.text.endswith("...") and len(hit.text) <= 166


def test_an_allowed_email_does_not_hide_a_private_one_after_it():
    line = "Co-Authored-By: Claude <noreply@anthropic.com>, Jane <jane@example.com>"
    (hit,) = export_public.scan_text("x.md", line, export_public.load_patterns(Path("missing")))
    assert hit.kind == "email address" and "jane@example.com" in hit.text
    assert kinds("Co-Authored-By: Claude <noreply@anthropic.com>") == []


@pytest.mark.parametrize(
    "line, kind",
    [
        ("mail JANE@EXAMPLE.COM", "email address"),
        ("see Claude.ai/code/session_abc", "claude.ai link"),
        ("CLAUDE.AI/chat/abc", "claude.ai link"),
        ("claude-session: https://example.com", "session trailer"),
        ("logged as s:0cabffc0 in the LOG", "session tag"),
        ("(s:ABCDEF12)", "session tag"),
    ],
)
def test_text_that_can_be_written_in_any_case_is_flagged_in_capitals(line, kind):
    assert kinds(line) == [kind]


@pytest.mark.parametrize(
    "line", ["a ~user typo", "this:0cabffc0", "s:0cabffc0a", "kinds: abcdefgh", "x~/y/", "?~\\X?", ">~/U?a"]
)
def test_near_misses_of_the_new_patterns_are_not_flagged(line):
    assert kinds(line) == []


@pytest.mark.parametrize(
    "line, flagged",
    [
        ("{ name = 'Steven Rice', email = 'srice@datarealminc.com' }", False),  # the starter's own credit
        ("Co-Authored-By: Claude <noreply@anthropic.com>", False),
        ("12345+someone@users.noreply.github.com", False),
        ("mail jane.noreply@example.com", True),  # contains "noreply" but is a personal address
        ("mail noreply@example.com", True),  # a noreply address that is not on the list
    ],
)
def test_allowed_emails_are_matched_whole(line, flagged):
    assert kinds(line) == (["email address"] if flagged else [])


def test_a_binary_files_readable_strings_are_scanned_with_their_byte_offset():
    data = b"\x00\x01PXR-USDC\x00\x00asset @://hostname.local/Users/someone/junk.usd@\x00" + "C:\\Users\\x".encode(
        "utf-16-le"
    )
    hits = export_public.scan_binary("assets/x.usd", data, export_public.load_patterns(Path("missing")))
    assert [(h.kind, h.binary) for h in hits] == [("home path", True), ("home path", True)]
    assert hits[0].line == data.index(b"asset") and "/Users/someone" in hits[0].text


def tar_of(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def fake_archive(monkeypatch, files):
    archive = tar_of(files)
    monkeypatch.setattr(
        export_public.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=archive)
    )


def test_export_scans_a_notebook_and_a_binary_files_strings(tmp_path, monkeypatch):
    """The whole export, not only text_of: a notebook's home path is a hit, the binary file is listed, and a home path
    in its readable strings is reported apart from the text hits."""
    files = {
        "notebook.ipynb": b'{"outputs": ["C:\\\\Users\\\\someone"]}',
        "media/frame.png": b"\x89PNG\r\n\x1a\n\x00\x00",
        "assets/robot.usd": b"\x00\x00ref @://hostname.local/Users/someone/junk.usd@\x00",
        "README.md": b"plain line about the robot",
        "Launch-Presentation.pdf": b"%PDF C:\\Users\\someone",
    }
    fake_archive(monkeypatch, files)
    out = tmp_path / "public"
    written, left_out, hits, binary = export_public.export(
        out, force=False, patterns=export_public.load_patterns(Path("missing"))
    )
    assert sorted(written) == ["README.md", "assets/robot.usd", "media/frame.png", "notebook.ipynb"]
    assert list(left_out) == ["Launch-Presentation.pdf"]
    assert sorted((h.path, h.kind, h.binary) for h in hits) == [
        ("assets/robot.usd", "home path", True),
        ("notebook.ipynb", "home path", False),
    ]
    assert sorted(binary) == ["assets/robot.usd", "media/frame.png"]
    assert (out / "notebook.ipynb").read_bytes() == files["notebook.ipynb"]
    text = export_public.report(out, written, left_out, hits, "abc1234", binary)
    assert "## Binary files: only their readable strings were scanned (2)" in text
    assert "## Lines that look private (1)" in text and "- `notebook.ipynb:1` (home path):" in text
    assert (
        "## Strings in binary files that look private (1)" in text
        and "- `assets/robot.usd` byte 2 (home path):" in text
    )


def test_force_never_clears_a_folder_the_script_did_not_make(tmp_path):
    out = tmp_path / "someone-elses"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(SystemExit, match="not touching it"):
        export_public.export(out, force=True, patterns=[])
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_force_refuses_an_export_that_holds_a_git_repo(tmp_path, monkeypatch):
    """The public repo may be made in the export folder; --force must not delete its history."""
    fake_archive(monkeypatch, {"README.md": b"robot"})
    out = tmp_path / "public"
    export_public.export(out, force=False, patterns=[])
    (out / ".git").mkdir()
    (out / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    with pytest.raises(SystemExit, match="git repo"):
        export_public.export(out, force=True, patterns=[])
    assert (out / ".git" / "HEAD").exists()


def test_a_failed_rebuild_leaves_the_last_export_whole(tmp_path, monkeypatch):
    fake_archive(monkeypatch, {"README.md": b"first"})
    out = tmp_path / "public"
    export_public.export(out, force=False, patterns=[])

    def archive_fails(*a, **k):
        raise subprocess.CalledProcessError(128, a[0])

    monkeypatch.setattr(export_public.subprocess, "run", archive_fails)
    with pytest.raises(subprocess.CalledProcessError):
        export_public.export(out, force=True, patterns=[])
    assert (out / "README.md").read_bytes() == b"first"


def test_force_rebuilds_and_clears_a_leftover_build_folder(tmp_path, monkeypatch):
    fake_archive(monkeypatch, {"README.md": b"first"})
    out = tmp_path / "public"
    export_public.export(out, force=False, patterns=[])
    leftover = tmp_path / ".public.building"
    leftover.mkdir()
    (leftover / export_public.MARKER).write_text("x", encoding="utf-8")
    fake_archive(monkeypatch, {"README.md": b"second"})
    export_public.export(out, force=True, patterns=[])
    assert (out / "README.md").read_bytes() == b"second" and (out / export_public.MARKER).exists()
    assert not leftover.exists()


def test_a_locked_export_folder_is_left_whole(tmp_path, monkeypatch):
    """Windows refuses to rename a folder a shell is cd'd into; the old export must survive that, marker and all, so
    the next --force still accepts it."""
    fake_archive(monkeypatch, {"README.md": b"first"})
    out = tmp_path / "public"
    export_public.export(out, force=False, patterns=[])
    real_rename = Path.rename

    def rename(self, target):
        if self == out:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    fake_archive(monkeypatch, {"README.md": b"second"})
    with pytest.raises(PermissionError):
        export_public.export(out, force=True, patterns=[])
    assert (out / "README.md").read_bytes() == b"first" and (out / export_public.MARKER).exists()
    monkeypatch.setattr(Path, "rename", real_rename)
    export_public.export(out, force=True, patterns=[])  # once unlocked, the next run takes it and the leftover build
    assert (out / "README.md").read_bytes() == b"second"
    assert not (tmp_path / ".public.building").exists() and not (tmp_path / ".public.old").exists()
