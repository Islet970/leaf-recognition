"""pic/ 数据看板扫描服务（plans.md §5.7）。

- 扫描项目根 ``pic/`` 的一级子目录，目录与文件均按名称排序；
- 忽略 ``~$*``（Office 临时文件）与隐藏文件/目录；
- 图片（.png/.jpg/.jpeg）返回 URL 与大小；
- CSV 返回列名、前 200 行与总行数；
- .docx/.pdf 返回下载 URL。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DOCUMENT_SUFFIXES = {".docx", ".pdf"}
CSV_PREVIEW_ROWS = 200
CSV_MAX_BYTES = 1024 * 1024  # 超过 1 MB 的 CSV 不做解析，仅提供下载


def _encode_url(*parts: str) -> str:
    """生成 /pic/<dir>/<file> URL，POSIX 斜杠并逐段 URL 编码。"""
    encoded = "/".join(quote(str(part), safe="") for part in parts)
    return f"/pic/{encoded}"


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _is_office_temp(name: str) -> bool:
    return name.startswith("~$")


class GalleryService:
    """扫描 pic/ 并提供看板所需的文件树。"""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.pic_root = self.project_root / "pic"

    # ------------------------------------------------------------------ 对外
    def list_gallery(self) -> dict[str, Any]:
        directories: list[dict[str, Any]] = []
        if not self.pic_root.is_dir():
            return {"directories": directories}

        subdirs = sorted(
            (p for p in self.pic_root.iterdir() if p.is_dir() and not _is_hidden(p.name)),
            key=lambda p: p.name,
        )
        for subdir in subdirs:
            files = self._scan_directory(subdir)
            directories.append({"name": subdir.name, "files": files})
        return {"directories": directories}

    # ------------------------------------------------------------------ 内部
    def _scan_directory(self, directory: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            candidates = list(directory.iterdir())
        except OSError as exc:
            print(f"[GalleryService] 无法读取 {directory}: {exc}", file=sys.stderr)
            return entries

        candidates = [
            p
            for p in candidates
            if p.is_file() and not _is_hidden(p.name) and not _is_office_temp(p.name)
        ]
        candidates.sort(key=lambda p: p.name)

        for path in candidates:
            entry = self._describe_file(directory.name, path)
            if entry is not None:
                entries.append(entry)
        return entries

    def _describe_file(self, directory_name: str, path: Path) -> dict[str, Any] | None:
        suffix = path.suffix.lower()
        try:
            stat = path.stat()
        except OSError:
            return None

        base: dict[str, Any] = {
            "name": path.name,
            "url": _encode_url(directory_name, path.name),
            "size": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }

        if suffix in IMAGE_SUFFIXES:
            base["type"] = "image"
            return base

        if suffix == ".csv":
            base.update(self._describe_csv(path, int(stat.st_size)))
            return base

        if suffix in DOCUMENT_SUFFIXES:
            base["type"] = "file"
            return base

        # 其他扩展名不进入看板
        return None

    def _describe_csv(self, path: Path, size: int) -> dict[str, Any]:
        """解析 CSV；失败时降级为可下载文件并附 error 字段。"""
        if size > CSV_MAX_BYTES:
            return {
                "type": "file",
                "error": f"CSV 文件超过 {CSV_MAX_BYTES // 1024} KB，未解析",
            }

        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "gbk"):
            try:
                # 只读取前 200 行做预览，总行数用 csv.reader 流式统计
                head = pd.read_csv(
                    path, encoding=encoding, nrows=CSV_PREVIEW_ROWS
                )
                n_rows = self._count_csv_rows(path, encoding)
                # to_json 会把 NaN 转成 null，且 numpy 标量转为 JSON 原生类型
                rows = json.loads(head.to_json(orient="records", force_ascii=False))
                return {
                    "type": "csv",
                    "columns": [str(c) for c in head.columns],
                    "rows": rows,
                    "n_rows": n_rows,
                }
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break

        print(f"[GalleryService] CSV 解析失败 {path}: {last_error}", file=sys.stderr)
        return {
            "type": "file",
            "error": f"CSV 解析失败: {last_error}",
        }

    @staticmethod
    def _count_csv_rows(path: Path, encoding: str) -> int:
        """按 CSV 语法统计数据行数（不含表头）。"""
        with path.open("r", encoding=encoding, errors="replace", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            return sum(1 for _ in reader)
