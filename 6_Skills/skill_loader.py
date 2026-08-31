"""
skill_loader.py
===============

The filesystem side of the Skills pattern. It reads skills from disk the way real
skill systems do — each skill is a folder with a SKILL.md (frontmatter + body) and a
bundled script — and it exposes the THREE LEVELS of progressive disclosure:

  Level 1  discover_skills()  -> read ONLY frontmatter (name + description) of each SKILL.md
  Level 2  load_skill_body()  -> read the full instructions for ONE skill, on demand
  Level 3  run_script()       -> execute that skill's bundled script (the real capability)

Nothing here calls an LLM. It is plain file + subprocess plumbing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into its frontmatter dict and its body. Frontmatter is the
    block between the first two '---' lines."""
    if text.lstrip().startswith("---"):
        _, fm, body = text.split("---", 2)
        meta = {}
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body.strip()
    return {}, text.strip()


def discover_skills(skills_dir: Path = SKILLS_DIR) -> dict:
    """LEVEL 1 — the cheap table of contents. Reads only the frontmatter of each
    SKILL.md, never the body. Returns name -> {description, dir}."""
    skills = {}
    for md in sorted(skills_dir.glob("*/SKILL.md")):
        meta, _ = _parse_frontmatter(md.read_text(encoding="utf-8"))
        name = meta.get("name", md.parent.name)
        skills[name] = {"description": meta.get("description", ""), "dir": md.parent}
    return skills


def load_skill_body(skill: dict) -> str:
    """LEVEL 2 — the full instructions for one skill, loaded only when it is chosen."""
    _, body = _parse_frontmatter((skill["dir"] / "SKILL.md").read_text(encoding="utf-8"))
    return body


def _script_path(skill: dict) -> Path | None:
    scripts = sorted(p for p in skill["dir"].glob("*.py"))
    return scripts[0] if scripts else None


def run_script(skill: dict, args: dict) -> dict:
    """LEVEL 3 — execute the skill's bundled script with CLI args, return parsed JSON.
    Each key/value in `args` becomes `--key value`."""
    script = _script_path(skill)
    if script is None:
        return {"error": "no script bundled with this skill"}
    argv = [sys.executable, str(script)]
    for k, v in args.items():
        argv += [f"--{k}", str(v)]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {"error": "script timed out"}
    if out.returncode != 0:
        return {"error": (out.stderr or "script failed").strip()[:400]}
    try:
        return json.loads(out.stdout.strip())
    except json.JSONDecodeError:
        return {"raw": out.stdout.strip()[:400]}