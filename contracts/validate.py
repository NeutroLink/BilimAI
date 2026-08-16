#!/usr/bin/env python
"""Validate every example in contracts/examples against its schema (request + response).

Usage: python contracts/validate.py            → prints OK / errors, exit 1 on any failure
       python contracts/validate.py <file>     → validate one JSON (must have request/response)
Also usable as a library: validate_request(obj), validate_response(obj).
"""
import json, sys
from pathlib import Path
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

HERE = Path(__file__).resolve().parent
TYPES = ["dictation", "math", "mcq", "retelling", "open_question"]


def _registry():
    reg = Registry()
    for p in HERE.glob("*.schema.json"):
        s = json.load(open(p, encoding="utf-8"))
        reg = reg.with_resource(s["$id"], Resource.from_contents(s))
        reg = reg.with_resource(p.name, Resource.from_contents(s))          # relative refs like "common.schema.json#/..."
    return reg


_REG = _registry()


def _validator(atype, part):
    schema = json.load(open(HERE / f"{atype}.schema.json", encoding="utf-8"))
    sub = {"$schema": schema["$schema"], "$id": schema["$id"] + f"#{part}", "$ref": f"{atype}.schema.json#/$defs/{part}"}
    return Draft202012Validator(sub, registry=_REG)


def validate_request(obj):
    return sorted(_validator(obj["type"], "Request").iter_errors(obj), key=lambda e: e.path)


def validate_response(obj):
    return sorted(_validator(obj["type"], "Response").iter_errors(obj), key=lambda e: e.path)


def main(paths):
    ok = True
    for p in paths:
        ex = json.load(open(p, encoding="utf-8"))
        for part, fn in (("request", validate_request), ("response", validate_response)):
            errs = fn(ex[part])
            if errs:
                ok = False
                print(f"FAIL {Path(p).name} [{part}]")
                for e in errs[:8]:
                    print(f"   - {'/'.join(map(str, e.path)) or '<root>'}: {e.message[:160]}")
            else:
                print(f"ok   {Path(p).name} [{part}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(str(p) for p in (HERE / "examples").glob("*.json")))
