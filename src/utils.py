"""通用工具：随机种子、日志、指标、checkpoint。

本模块不依赖项目内其他模块，可被训练 / 评测 / 网页共同复用。
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，尽量保证可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN 仅在 GPU 上生效；CPU 训练时设置也无副作用。
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 视算子兼容性启用确定性算法；不兼容时只警告，不中断训练。
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError, RuntimeError):
        pass


class _Utf8ConsoleHandler(logging.StreamHandler):
    """控制台 handler：输出被重定向/管道时直接写 UTF-8 字节。

    Windows 下 ``sys.stdout.encoding`` 常为 GBK，中文日志经管道会乱码；
    交互式终端保持原编码（cmd 里中文本身可正常显示）。
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = self.stream
        try:
            msg = self.format(record) + self.terminator
            encoding = (getattr(stream, "encoding", "") or "").replace("-", "").lower()
            is_tty = bool(getattr(stream, "isatty", lambda: False)())
            if encoding != "utf8" and not is_tty and hasattr(stream, "buffer"):
                stream.flush()  # 先输出 text 层已缓冲的内容，保持日志顺序
                stream.buffer.write(msg.encode("utf-8", errors="replace"))
                stream.buffer.flush()
            else:
                stream.write(msg)
                stream.flush()
        except Exception:  # noqa: BLE001 - logging 约定：交给 handleError
            self.handleError(record)


def get_logger(name: str, log_file: Path | str | None = None) -> logging.Logger:
    """返回同时输出到控制台（和可选文件）的 logger。

    - 文件以 UTF-8 写入；
    - 重复调用不会叠加 handler（旧的同类 handler 会先被移除）。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 移除本函数之前创建的 handler，避免重复输出。
    for handler in list(logger.handlers):
        if getattr(handler, "_leaf_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = _Utf8ConsoleHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console._leaf_handler = True  # type: ignore[attr-defined]
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._leaf_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """维护一个滑动累计的平均值（loss / accuracy 等）。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)
        n = int(n)
        self.val = value
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.4f}, count={self.count})"


def save_checkpoint(path: Path | str, state: Mapping[str, Any]) -> None:
    """保存 checkpoint（父目录不存在时自动创建）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), path)


def load_checkpoint(
    path: Path | str, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """加载 checkpoint；checkpoint 含配置/类别名等 Python 对象，故 weights_only=False。"""
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # 兼容不支持 weights_only 的旧版 PyTorch
        obj = torch.load(path, map_location=map_location)
    return obj


def macro_f1(y_true, y_pred) -> float:
    """macro-F1；空输入返回 0.0，避免 sklearn 抛错。"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量；默认统计全部参数。"""
    if trainable_only:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in model.parameters()))
