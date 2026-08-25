"""Deploy this project to a Hugging Face Space.

    # once, interactively - the token is stored by the CLI, never passed here
    .venv/Scripts/hf.exe auth login

    python scripts/deploy_hf.py --space <your-username>/oil-spill-detection

Creates the Space if it does not exist, then uploads exactly the files the
container needs. Data and the virtualenv are excluded - the image generates
its demo scenes and downloads the incident registry at build time, so the
Space repo stays small and the build stays reproducible.

The token is read from the CLI's stored credentials. It is never taken as an
argument, so it cannot end up in your shell history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything the container needs, and nothing else.
INCLUDE_DIRS = [
    "core", "ingest", "detect", "drift", "attribute", "decision",
    "api", "ui", "scripts", "configs",
]
INCLUDE_FILES = ["Dockerfile", "requirements.txt", "PROJECT.md", "DEPLOY.md"]

IGNORE = [
    "**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**",
    "data/**", "runs/**", "models/**", ".venv/**", "tests/**",
    "**/.DS_Store",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", required=True,
                    help="target Space, e.g. yourname/oil-spill-detection")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private (default public)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and stop")
    args = ap.parse_args()

    if "/" not in args.space:
        raise SystemExit("--space must be <username>/<space-name>")

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        raise SystemExit(
            "Not logged in to Hugging Face. Run this first:\n"
            "    .venv/Scripts/hf.exe auth login\n"
            "Create a token with WRITE access at "
            "https://huggingface.co/settings/tokens"
        )
    print(f"Logged in as: {who.get('name')}")

    # The Space needs SPACE_README.md as its README.md - that file carries the
    # YAML front matter that tells Spaces to use Docker and which port to open.
    space_readme = REPO_ROOT / "SPACE_README.md"
    if not space_readme.exists():
        raise SystemExit("SPACE_README.md is missing - it carries the Space front matter")

    staged: list[tuple[Path, str]] = [(space_readme, "README.md")]
    for name in INCLUDE_FILES:
        path = REPO_ROOT / name
        if path.exists():
            staged.append((path, name))
    for directory in INCLUDE_DIRS:
        base = REPO_ROOT / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            staged.append((path, rel))

    total = sum(p.stat().st_size for p, _ in staged)
    print(f"Staged {len(staged)} files, {total/1e6:.1f} MB")
    if args.dry_run:
        for _, rel in sorted(staged, key=lambda s: s[1])[:25]:
            print("   ", rel)
        print(f"    ... ({len(staged)} total)")
        return 0

    print(f"Ensuring Space {args.space} exists ...")
    try:
        api.create_repo(
            repo_id=args.space, repo_type="space", space_sdk="docker",
            private=args.private, exist_ok=True,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(f"Could not create the Space: {exc}")

    print("Uploading ...")
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(REPO_ROOT),
        allow_patterns=[rel for _, rel in staged],
        ignore_patterns=IGNORE,
        commit_message="Deploy oil spill detection and vessel attribution",
    )
    # README has to be uploaded under its new name, which allow_patterns above
    # cannot express since the source file is called SPACE_README.md.
    api.upload_file(
        path_or_fileobj=str(space_readme),
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
        commit_message="Space front matter",
    )

    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nDeployed: {url}")
    print("The first Docker build takes about 10 minutes. Watch the Logs tab.")
    print("If the build fails, the log names the failing step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
