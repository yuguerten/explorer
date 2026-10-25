from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Optional

from .models import SourceFile, SourceReference


def _normalize_symbol(sym: object) -> tuple[Optional[str], Optional[str]]:
    """Return (class_name, method_name) for a module-level symbol.

    Handles classes, functions, and bound/unbound methods (which present a
    class-qualified __qualname__ such as 'GPT2Model.forward').
    """
    qualname = getattr(sym, "__qualname__", None)
    if not qualname:
        qualname = getattr(sym, "__name__", None)
    if not qualname:
        return None, None
    parts = qualname.split(".")
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


def collect_source_file(sym: object, package_version: Optional[str] = None) -> Optional[SourceFile]:
    """Resolve a symbol's defining file and return a SourceFile with content."""
    try:
        path = inspect.getsourcefile(sym)
    except (TypeError, OSError):
        return None
    if not path:
        return None
    path = str(Path(path).resolve())
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return None
    display = Path(path).name
    file_id = _file_id(path, display)
    return SourceFile(
        file_id=file_id,
        path=path,
        display_name=display,
        content=content,
        package_version=package_version,
    )


def _file_id(path: str, display: str) -> str:
    import hashlib
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"file:{display}:{digest}"


def collect_symbol_reference(sym: object, file: SourceFile, origin: str = "inspected-symbol") -> Optional[SourceReference]:
    """Extract a SourceReference (class or method level) for a symbol."""
    try:
        lines, start = inspect.getsourcelines(sym)
    except (TypeError, OSError, IOError):
        return None
    end = start + len(lines) - 1
    cls, method = _normalize_symbol(sym)
    granularity = "method" if method else "class"
    return SourceReference(
        id=_ref_id(cls, method, file.file_id),
        source_file_id=file.file_id,
        symbol=cls or "",
        method=method,
        start_line=start,
        end_line=end,
        granularity=granularity,
        architecture_node_ids=[],
        origin=origin,
    )


def _ref_id(cls: str, method: Optional[str], file_id: str) -> str:
    import hashlib
    raw = f"{cls}.{method}@{file_id}"
    return "ref:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _ast_class_body(module: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def method_line_range(module: ast.Module, class_name: str, method_name: str) -> Optional[tuple[int, int]]:
    """Return (start, end) line range for a method inside a class, via AST.

    Falls back to scanning source text when the method is defined dynamically
    or the AST node cannot be located.
    """
    cls = _ast_class_body(module, class_name)
    if cls is None:
        return None
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node.lineno, node.end_lineno
    return None
