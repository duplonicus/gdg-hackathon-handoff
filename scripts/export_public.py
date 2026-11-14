"""Build the public copy of the project in a separate folder, and scan it for anything private. Local only.

The public project link (a Kaggle submission part) is meant to be a fresh repository, not this one: this repo's history
carries working notes and session links. This script takes the committed tree at HEAD (``git archive``, so nothing
untracked or ignored can leak), leaves out what should not be redistributed, and writes a scan report of every line
that looks private. It publishes nothing and makes no commit: what goes on GitHub, and when, is dup's call.

    .\\run.ps1 scripts/export_public.py                     # to ..\\GDG-Hackathon-public, report in output/
    .\\run.ps1 scripts/export_public.py --out D:\\somewhere --force

The private words to look for (names, account handles) live in ``config/public_scan.local.txt``, one regex per line,
which git ignores; ``config/public_scan.example.txt`` shows the shape. Generic patterns (home paths, keys, chat links)
are built in.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ".export-public"  # written into the export; --force only ever clears a folder that carries it

# Left out of the public copy, with the reason the report prints.
EXCLUDE: dict[str, str] = {
    "Launch-Presentation.pdf": "the host's launch deck, not this project's to redistribute",
    "media/ladder-test/": "an old style test with the retired ladder's numbers",
}

# Generic signs of private content; the local file adds names and handles. A path separator is any run of slashes or
# backslashes, so a path matches however it is spelled: C:\Users in a doc, C:\\Users in a Python or JSON string, C:/Users,
# and /mnt/c/Users from WSL.
SEP = r"[\\/]+"
# A drive's colon: any letter may come before it (a colour code's "m" in "\x1b[0;32mC:\Users", the "n" of an escaped
# "\nC:\\Users", which a word boundary refused), but not "//" after it, which is a URL's scheme (https://ollama, gs://models),
# unless "users" follows the "//" (C://Users is a path).
COLON = r":(?!//(?!users))"
BUILTIN: list[tuple[str, str]] = [
    # A Windows or WSL home, a Linux home, a drive-less /Users/name (macOS, or a URL that carries one, as MH3.usd does),
    # and ~/ or ~\ (a home written short).
    (
        "home path",
        (
            rf"(?i)(?:[a-z]{COLON}{SEP}users{SEP}|{SEP}mnt{SEP}[a-z]{SEP}users{SEP}|{SEP}home{SEP}[a-z]|{SEP}users{SEP}[a-z]"
            rf"|(?<![\w~])~{SEP}[\w.-]+{SEP})"
        ),
    ),
    ("local drive path", rf"(?i)(?:[d-z]{COLON}|{SEP}mnt{SEP}[d-z]){SEP}(?:models|toolkits|ollama)"),
    ("claude.ai link", r"(?i)claude\.ai/(?:code|artifact|chat)"),
    ("session trailer", r"(?i)Claude-Session:"),
    ("session tag", r"(?i)(?<![a-z0-9])s:[0-9a-f]{8}(?![a-z0-9])"),  # the LOG's "s:0cabffc0" form
    ("email address", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("api key", r"(?:sk-ant-|sk-|AIza|ghp_|gho_|hf_|ops_)[A-Za-z0-9._-]{20,}"),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
# Addresses that are fine to publish, each matched as the whole address: the starter kit's own author credit
# (pyproject.toml), Claude's noreply trailer, and GitHub's per-user noreply addresses. Nothing else, so a personal
# "jane.noreply@..." is still flagged.
ALLOWED_EMAILS = {"srice@datarealminc.com", "noreply@anthropic.com"}
GITHUB_NOREPLY = re.compile(r"(?i)[a-z0-9._+-]+@users\.noreply\.github\.com")


def allowed_email(address: str) -> bool:
    return address.lower() in ALLOWED_EMAILS or GITHUB_NOREPLY.fullmatch(address) is not None


@dataclass(frozen=True)
class Hit:
    path: str
    line: int  # the line number, or for a binary file the byte offset of the readable string
    kind: str
    text: str
    binary: bool = False


def text_of(data: bytes) -> str | None:
    """A file's text to scan, or None for a binary file, judged by its bytes rather than its suffix (going by suffix
    skipped the notebook, .ipynb, the one file made for publishing). UTF-32 and UTF-16 with a byte-order mark are decoded
    as such, UTF-32 first since its little-endian mark starts with UTF-16's (read as UTF-16 it puts a NUL after every
    character, so no pattern could match); anything else with a NUL byte is binary; the rest is read as UTF-8, a byte
    that is not UTF-8 (a cp1252 quote) turning into a replacement character instead of costing the file its scan."""
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


# Runs of readable text inside a binary file, as `strings` finds them: ASCII and UTF-16-LE, four characters or more.
# Compressed content (PNG image data, most PDF streams, video) is not readable this way.
ASCII_RUN = re.compile(rb"[\x20-\x7e]{4,}")
UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")


def readable_strings(data: bytes) -> list[tuple[int, str]]:
    """(byte offset, text) for each readable run in ``data``."""
    runs = [(m.start(), m.group().decode("ascii")) for m in ASCII_RUN.finditer(data)]
    runs += [(m.start(), m.group().decode("utf-16-le")) for m in UTF16_RUN.finditer(data)]
    return sorted(runs)


def scan_binary(path: str, data: bytes, patterns: list[tuple[str, re.Pattern[str]]]) -> list[Hit]:
    """The text patterns over a binary file's readable strings; each hit gives the string's byte offset."""
    hits = []
    for offset, run in readable_strings(data):
        for hit in scan_text(path, run, patterns):
            hits.append(Hit(path, offset, hit.kind, hit.text, binary=True))
    return hits


def excluded(path: str) -> str | None:
    """The reason ``path`` stays out of the public copy, or None when it goes in."""
    for rule, why in EXCLUDE.items():
        if path == rule or (rule.endswith("/") and path.startswith(rule)):
            return why
    return None


def load_patterns(local: Path) -> list[tuple[str, re.Pattern[str]]]:
    """The built-in patterns plus one "private word" pattern per non-comment line of ``local`` (if it exists)."""
    patterns = [(kind, re.compile(rx)) for kind, rx in BUILTIN]
    if local.exists():
        for raw in local.read_text(encoding="utf-8").splitlines():
            rx = raw.strip()
            if rx and not rx.startswith("#"):
                patterns.append(("private word", re.compile(rx)))
    return patterns


def excerpt(line: str, start: int, width: int = 160) -> str:
    """Up to ``width`` characters of ``line`` around ``start``, so the quoted text holds the match even far along a long
    line; "..." marks a cut."""
    lead = max(0, start - width // 4)
    cut = line[lead : lead + width]
    return ("..." if lead else "") + cut.strip() + ("..." if lead + width < len(line.rstrip()) else "")


def scan_text(path: str, text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[Hit]:
    """One hit per pattern per line, for its first match that is not an allowed email address (a noreply address first
    on a line must not hide a real one after it)."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        for kind, rx in patterns:
            for match in rx.finditer(line):
                if kind == "email address" and allowed_email(match.group(0)):
                    continue
                hits.append(Hit(path, number, kind, excerpt(line, match.start())))
                break
    return hits


def export(
    out: Path, *, force: bool, patterns: list[tuple[str, re.Pattern[str]]]
) -> tuple[list[str], dict[str, str], list[Hit], list[str]]:
    """Write HEAD's tree minus EXCLUDE to ``out``; return (files written, files left out with why, scan hits, binary
    files, which are scanned by their readable strings only).

    The copy is built in a sibling folder and swapped in only once it is whole, so a failed ``git archive`` or write
    leaves the last export as it was. The swap renames the old export aside before anything is deleted: a folder that
    Windows holds open (a shell cd'd into it) refuses the rename, and the old export stays whole."""
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to rebuild it")
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this script (no {MARKER}); not touching it")
        if (out / ".git").exists():
            raise SystemExit(f"{out} holds a git repo, which --force would delete; export to another --out")
    tar_bytes = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    build = out.with_name(f".{out.name}.building")
    old = out.with_name(f".{out.name}.old")
    for leftover in (build, old):
        if leftover.exists():
            if not (leftover / MARKER).exists():
                raise SystemExit(f"{leftover} exists and was not made by this script (no {MARKER}); not touching it")
            shutil.rmtree(leftover)
    build.mkdir(parents=True)
    (build / MARKER).write_text("made by scripts/export_public.py; safe for it to rebuild\n", encoding="utf-8")
    written, left_out, hits, binary = [], {}, [], []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            why = excluded(member.name)
            if why is not None:
                left_out[member.name] = why
                continue
            data = tar.extractfile(member).read()
            target = build / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written.append(member.name)
            text = text_of(data)
            if text is None:
                binary.append(member.name)
                hits += scan_binary(member.name, data, patterns)
            else:
                hits += scan_text(member.name, text, patterns)
    if out.exists():
        out.rename(old)
        try:
            build.rename(out)
        except OSError:
            old.rename(out)
            raise
        shutil.rmtree(old, ignore_errors=True)
    else:
        build.rename(out)
    return written, left_out, hits, binary


def report(
    out: Path, written: list[str], left_out: dict[str, str], hits: list[Hit], commit: str, binary: list[str] = ()
) -> str:
    text_hits = [h for h in hits if not h.binary]
    binary_hits = [h for h in hits if h.binary]
    lines = [
        f"# Public export scan ({commit})",
        "",
        f"Exported {len(written)} files to `{out}`. Nothing was published or committed.",
        "",
    ]
    lines += ["## Left out", ""] + [f"- `{p}`: {why}" for p, why in sorted(left_out.items())] + [""]
    lines += [f"## Binary files: only their readable strings were scanned ({len(binary)})", ""]
    lines += (
        ["Compressed content inside them (PNG image data, most PDF streams, video) was not read.", ""]
        + [f"- `{p}`" for p in sorted(binary)]
        if binary
        else ["None."]
    )
    lines += [""]
    lines += [f"## Lines that look private ({len(text_hits)})", ""]
    lines += (
        ["Fix each in the source repo, commit, and export again; or decide it is fine to publish.", ""]
        if text_hits
        else ["None."]
    )
    lines += [f"- `{h.path}:{h.line}` ({h.kind}): `{quoted(h.text)}`" for h in text_hits]
    lines += ["", f"## Strings in binary files that look private ({len(binary_hits)})", ""]
    lines += (
        [
            "Each gives the byte offset of the string. Not ours to edit in someone else's asset: decide or ask its owner.",
            "",
        ]
        if binary_hits
        else ["None."]
    )
    lines += [f"- `{h.path}` byte {h.line} ({h.kind}): `{quoted(h.text)}`" for h in binary_hits]
    return "\n".join(lines) + "\n"


def quoted(text: str) -> str:
    return text.replace("`", "'")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT.parent / f"{ROOT.name}-public")
    parser.add_argument("--report", type=Path, default=ROOT / "output" / "public-export-scan.md")
    parser.add_argument("--force", action="store_true", help="rebuild --out if this script made it before")
    args = parser.parse_args(argv)
    patterns = load_patterns(ROOT / "config" / "public_scan.local.txt")
    written, left_out, hits, binary = export(args.out, force=args.force, patterns=patterns)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(args.out, written, left_out, hits, commit, binary), encoding="utf-8")
    by_kind: dict[str, int] = {}
    for h in hits:
        by_kind[h.kind] = by_kind.get(h.kind, 0) + 1
    print(
        f"exported {len(written)} files from {commit} to {args.out}; left out {len(left_out)}; binary {len(binary)}; private-looking lines {len(hits)} {by_kind}"
    )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
