"""JSON / JSONL load/save helpers for pydantic BaseModel objects.

Ported from flashinfer_bench/data/json_utils.py.
"""

from pathlib import Path
from typing import List, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json_file(obj: BaseModel, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj.model_dump_json(indent=2, exclude_unset=True))


def load_json_file(model_cls: Type[T], path: Union[str, Path]) -> T:
    with open(Path(path), "r", encoding="utf-8") as f:
        return model_cls.model_validate_json(f.read())


def save_jsonl_file(objs: List[BaseModel], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        lines = [o.model_dump_json(indent=None) for o in objs]
        f.write("\n".join(lines) + "\n")


def load_jsonl_file(model_cls: Type[T], path: Union[str, Path]) -> List[T]:
    out: List[T] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model_cls.model_validate_json(line))
    return out


def append_jsonl_file(objs: List[BaseModel], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline_prefix = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            needs_newline_prefix = f.read(1) != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_newline_prefix:
            f.write("\n")
        lines = [o.model_dump_json(indent=None) for o in objs]
        f.write("\n".join(lines) + "\n")
