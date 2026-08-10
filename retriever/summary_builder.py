"""
Day 5a — Build embeddable text summaries from repo_map.json
For each function/method/class, produces a short text description
used as the thing we actually embed. Docstrings are used when present;
falls back to a signature-based description when they're missing.
"""

import json
from typing import Dict, Any, List, Optional


def build_function_summary(func: Dict[str, Any], filepath: str, class_name: Optional[str] = None) -> str:
    """Build embeddable text for a single function/method with safe parsing."""
    func_name = func.get("name", "unknown_function")
    qualified_name = f"{class_name}.{func_name}" if class_name else func_name

    # Safe argument extraction handling dicts, strings, or missing 'name' keys
    raw_args = func.get("args", [])
    arg_names = []
    for a in raw_args:
        if isinstance(a, dict):
            arg_names.append(a.get("name", "arg"))
        elif isinstance(a, str):
            arg_names.append(a)
        else:
            arg_names.append(str(a))

    signature = f"{qualified_name}({', '.join(arg_names)})"

    docstring = func.get("docstring")
    if docstring:
        return f"{signature}: {docstring.strip()}"
    else:
        return f"Function {signature} in {filepath}"


def build_class_summary(cls: Dict[str, Any], filepath: str) -> str:
    """Build embeddable text for a class as a whole."""
    class_name = cls.get("name", "UnknownClass")
    docstring = cls.get("docstring")
    if docstring:
        return f"Class {class_name}: {docstring.strip()}"
    return f"Class {class_name} in {filepath}"


def build_summary_units(repo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk the whole repo_map and produce one 'unit' per function/method/class:
    {id, text (for embedding), metadata (filepath, name, type, lineno)}
    """
    units = []
    modules = repo_map.get("modules", {}) if isinstance(repo_map, dict) else {}

    for filepath, file_data in modules.items():
        if not isinstance(file_data, dict):
            continue

        # Top-level functions
        for func in file_data.get("functions", []):
            func_name = func.get("name", "unknown")
            unit_id = f"{filepath}::{func_name}"
            units.append({
                "id": unit_id,
                "text": build_function_summary(func, filepath),
                "metadata": {
                    "filepath": filepath,
                    "name": func_name,
                    "type": "function",
                    "lineno": func.get("lineno", 1),
                },
            })

        # Classes + their methods
        for cls in file_data.get("classes", []):
            cls_name = cls.get("name", "UnknownClass")
            class_unit_id = f"{filepath}::{cls_name}"
            units.append({
                "id": class_unit_id,
                "text": build_class_summary(cls, filepath),
                "metadata": {
                    "filepath": filepath,
                    "name": cls_name,
                    "type": "class",
                    "lineno": cls.get("lineno", 1),
                },
            })

            for method in cls.get("methods", []):
                method_name = method.get("name", "unknown_method")
                unit_id = f"{filepath}::{cls_name}.{method_name}"
                units.append({
                    "id": unit_id,
                    "text": build_function_summary(method, filepath, class_name=cls_name),
                    "metadata": {
                        "filepath": filepath,
                        "name": f"{cls_name}.{method_name}",
                        "type": "method",
                        "lineno": method.get("lineno", 1),
                    },
                })

    return units


if __name__ == "__main__":
    try:
        with open("sample_repo/repo_map.json") as f:
            repo_map = json.load(f)

        units = build_summary_units(repo_map)
        print(f"Built {len(units)} summary units\n")

        for u in units[:6]:
            print(f"[{u['metadata']['type']}] {u['id']}")
            print(f"  -> {u['text']}\n")
    except FileNotFoundError:
        print("Error: sample_repo/repo_map.json not found.")