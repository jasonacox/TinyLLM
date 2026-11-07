"""requirements.txt must be installable and cover every third-party import in the chatbot app.
Run: pytest chatbot/tests -q"""
import ast
import re
import sys
from pathlib import Path

CHATBOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = CHATBOT / "requirements.txt"

# import name -> distribution name (where they differ)
IMPORT_TO_DIST = {
    "PIL": "pillow-heif",          # Pillow is installed as a dependency of pillow-heif
    "pillow_heif": "pillow-heif",
    "socketio": "python-socketio",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "weaviate": "weaviate-client",
    "multipart": "python-multipart",
}


def _requirements():
    reqs = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, 1)[0].lower()
        reqs.setdefault(name, []).append(line)
    return reqs


def _third_party_imports():
    stdlib = set(sys.stdlib_module_names)
    found = set()
    for path in [*(CHATBOT / "app").rglob("*.py"), CHATBOT / "run.py"]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for n in names:
                top = n.split(".")[0]
                if top not in stdlib and top != "app":
                    found.add(top)
    return found


def test_each_requirement_listed_once():
    dupes = {k: v for k, v in _requirements().items() if len(v) > 1}
    assert not dupes, f"conflicting/duplicate requirement lines: {dupes}"


def test_requirements_cover_app_imports():
    reqs = _requirements()
    missing = sorted(m for m in _third_party_imports() if IMPORT_TO_DIST.get(m, m).lower() not in reqs)
    assert not missing, f"imported by the chatbot but not in requirements.txt: {missing}"
