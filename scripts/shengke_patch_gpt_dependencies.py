"""Repair the missing deprecated aliases in the pinned LangSegment 0.2.0 wheel.

Run with the GPT-SoVITS environment's Python. No speech/model code is changed.
"""

import hashlib
import importlib.metadata
from pathlib import Path


def main():
    distribution = importlib.metadata.distribution("LangSegment")
    if distribution.version != "0.2.0":
        raise SystemExit("This repair only supports LangSegment 0.2.0")
    path = Path(distribution.locate_file("LangSegment/__init__.py"))
    original = path.read_text()
    marker = "# Shengke: restore deprecated aliases to the existing filter functions."
    if marker in original:
        print("LangSegment compatibility repair already applied")
        return
    broken = "printList,setLangfilters,getLangfilters,setfilters,getfilters"
    if broken not in original:
        raise SystemExit("Unexpected LangSegment package; refusing to modify it")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != "ab9f0d48f303d252ae063adedeed0e49458d9b3e74c494c0a510dd05fc08af6d":
        raise SystemExit("Unexpected LangSegment file hash; refusing to modify it")
    repaired = original.replace(broken, "printList,setfilters,getfilters", 1)
    repaired += f"\n{marker}\nsetLangfilters = setfilters\ngetLangfilters = getfilters\n"
    path.write_text(repaired)
    print(f"Repaired {path}; original SHA-256: {before}")
    import LangSegment

    assert LangSegment.setLangfilters is LangSegment.setfilters
    assert LangSegment.getLangfilters is LangSegment.getfilters
    print("LangSegment import and compatibility aliases verified")


if __name__ == "__main__":
    main()
