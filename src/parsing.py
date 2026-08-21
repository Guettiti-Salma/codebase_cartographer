"""
parsing.py
----------
Pure, deterministic helpers for understanding source files: no LLM calls,
no network access. Two jobs:
  1. spot files that look like an "entry point" (where a program starts)
  2. extract the list of local files a given file imports

We use regex instead of a real parser (like Python's `ast` module) because
we want ONE code path that handles multiple languages (Python, JS/TS).
It's less precise than a full AST, but for drawing a dependency diagram,
"good enough and fast" beats "perfect and slow" — and critically, it's
free and instant, unlike asking an LLM to read every file.

A note on posixpath vs os.path: every repo-relative path in this project
(file_path, importer_path, all_files, edges — the "logical" identifiers)
is stored using forward slashes, ALWAYS, regardless of the OS running the
code. os.path.join/os.path.normpath/os.sep are OS-native — on Windows
they produce backslashes, which would silently break every set-membership
check against all_files (which is forward-slash) and make edges look
different on Windows vs Linux/Mac for the exact same repo. posixpath
always uses "/" no matter the OS, so it's used here for that reason —
os.path is reserved for actual filesystem calls (open(), os.walk()),
which need real OS paths, not these logical ones.
"""

import os          # real filesystem calls only (open, os.walk) — see note above
import posixpath   # OS-independent path *string* manipulation — used everywhere else
import re          # the regex patterns that find import statements


# File extensions we know how to chunk/parse, mapped to a short language tag.
# Anything not in this dict still gets indexed as plain text, just never traced.
LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

# Directories we never want to walk into: build artifacts, dependencies,
# and version-control internals — none of this is "the project's" code.
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next"}

# Filenames that conventionally mark where a program starts running.
ENTRY_POINT_FILENAMES = {"main.py", "app.py", "manage.py", "index.js", "index.ts", "server.js"}

# Matches Python imports: both "import x.y" and "from x.y import z".
# Two capture groups because only one branch of the "from|import" fires per match.
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)

# Matches JS/TS imports: `import ... from '...'` and `require('...')`.
JS_IMPORT_RE = re.compile(r"""(?:import\s+.*?from\s+|require\()\s*['"]([^'"]+)['"]""")


def detect_language(file_path: str):
    _, ext = posixpath.splitext(file_path)   # "src/app.py" -> (".", "src/app", ".py") style split, we want ext
    return LANGUAGE_BY_EXT.get(ext)          # .get() returns None for extensions we don't recognize


def is_entry_point(repo_relative_path: str, repo_root: str) -> bool:
    filename = posixpath.basename(repo_relative_path)   # "src/main.py" -> "main.py" — works on forward-slash
                                                          # paths on every OS, unlike os.path.basename on Windows

    if filename in ENTRY_POINT_FILENAMES:              # easy case: matches a known convention
        return True

    if filename.endswith(".py"):                        # harder case: check for Python's main-guard idiom
        # os.path.join here is intentional: this builds a REAL filesystem path for
        # open() to use, and Windows accepts forward slashes in that role just fine —
        # this is the one place in this function doing actual I/O, not string logic
        full_path = os.path.join(repo_root, repo_relative_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:  # ignore bad bytes, don't crash
                content = f.read()
        except OSError:                                   # file might be unreadable / a broken symlink
            return False
        if "__name__" in content and "__main__" in content:   # cheap heuristic, not a full parse — good enough
            return True

    return False                                          # nothing matched — not an entry point


def extract_imports(file_path: str, content: str) -> list:
    language = detect_language(file_path)                 # decide which regex applies

    if language == "python":
        matches = PY_IMPORT_RE.findall(content)            # list of tuples like ("os.path", "") or ("", "sys")
        return [m[0] or m[1] for m in matches if (m[0] or m[1])]   # flatten to a plain list, drop the empty side

    if language in ("javascript", "typescript"):
        return JS_IMPORT_RE.findall(content)                # findall already returns the captured path directly

    return []                                                # unknown language — nothing to trace


def resolve_import(importer_path: str, imported: str, repo_root: str, all_files: set):
    """
    Turn a raw import string ("./utils" or "myapp.helpers") into an actual
    repo-relative file path IF it points at a file we indexed. Returns
    None for third-party/stdlib imports, which we deliberately don't trace.
    """
    importer_dir = posixpath.dirname(importer_path)          # the folder the importing file lives in

    # --- relative JS/TS-style import: "./foo" or "../bar" ---
    if imported.startswith("."):
        candidate_base = posixpath.normpath(posixpath.join(importer_dir, imported))  # collapse "../" etc.,
                                                                                        # always forward-slash output
        for ext in (".js", ".jsx", ".ts", ".tsx"):           # try each plausible extension
            candidate = candidate_base + ext
            if candidate in all_files:
                return candidate
            index_candidate = posixpath.join(candidate_base, "index" + ext)   # "./components" -> ".../index.js"
            if index_candidate in all_files:
                return index_candidate
        return None                                           # didn't resolve to a file we actually indexed

    # --- dotted Python-style import: "myapp.helpers.utils" ---
    as_path = imported.replace(".", "/") + ".py"               # "myapp.helpers" -> "myapp/helpers.py"
    if as_path in all_files:
        return as_path
    parent_path = posixpath.dirname(as_path) + ".py"          # "myapp.helpers.util_fn" -> "myapp/helpers.py"
    if parent_path in all_files:
        return parent_path

    # --- package import: "import schedule" where schedule/ is a package (has __init__.py) ---
    # extremely common — any "import <package>" for a package rather than a single module
    # needs this check, or it silently fails to resolve every time
    package_init = imported.replace(".", "/") + "/__init__.py"
    if package_init in all_files:
        return package_init

    # --- src-layout fallback: package lives under src/, e.g. src/myapp/helpers.py ---
    # ("src layout" is a common Python packaging convention used by setuptools templates,
    # PyPA's own sample project, and plenty of real-world packages)
    src_as_path = "src/" + as_path
    if src_as_path in all_files:
        return src_as_path
    src_package_init = "src/" + imported.replace(".", "/") + "/__init__.py"
    if src_package_init in all_files:
        return src_package_init

    return None                                                 # probably a third-party or stdlib import