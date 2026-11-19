"""Build a local source-only archive from an explicit allowlist."""

import argparse
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".env.example", ".gitignore", "README.md", "pyproject.toml",
    "run_experiment.py", "test_llm.py", "Benchmark/__init__.py",
    "docs/benchmark_protocol.md", "examples/tictactoe_demo_group.json",
    "scripts/export_release.py",
)
SECRET_PATTERNS = (
    re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{18,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(rb"""(?i)\b(?:api_key|access_token|client_secret)\s*=\s*["'][^"'\r\n]{12,}["']"""),
)


def release_files(root: Path = ROOT) -> list[Path]:
    files = {root / name for name in FILES}
    files.update((root / "Bench").glob("*.py"))
    for name in ("langpsro", "Benchmark/screening"):
        files.update((root / name).rglob("*.py"))
    files.update((root / "tests").glob("test_*.py"))
    if (root / "LICENSE").is_file():
        files.add(root / "LICENSE")
    for path in files:
        rel = path.relative_to(root)
        if any((root / p).is_symlink() for p in (rel, *rel.parents)):
            raise ValueError(f"Symbolic links are not allowed in the release: {rel}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(files)


def create_archive(output: Path) -> int:
    contents = {}
    for path in release_files():
        rel = path.relative_to(ROOT)
        payload = path.read_bytes()
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Possible credential in {rel}; export stopped.")
        contents[rel.as_posix()] = payload

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        working_path = Path(tmp.name)
    try:
        with ZipFile(working_path, "w", compression=ZIP_DEFLATED) as archive:
            for name, payload in contents.items():
                archive.writestr(f"when-memory-matters/{name}", payload)
        os.replace(working_path, output)
    finally:
        working_path.unlink(missing_ok=True)
    return len(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/when-memory-matters-code.zip")
    args = parser.parse_args()
    try:
        count = create_archive(args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Exported {count} files to {args.output}")


if __name__ == "__main__":
    main()
