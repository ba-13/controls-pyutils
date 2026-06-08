import numpy as np
import matplotlib.pyplot as plt
from contextvars import ContextVar

_current_logger = ContextVar("current_logger", default=None)


class SimLogger:
    def __init__(self):
        self.data = {}
        self.current = {}

    def record(self, **signals):
        self.current.update(signals)

    def next_step(self):
        for name, value in self.current.items():
            self.data.setdefault(name, []).append(np.array(value).copy())
        self.current.clear()

    def array(self, name):
        return np.array(self.data[name])

    def plot(self, *names, row_size=4, col_size=3):
        t = self.array("t")
        N = len(names)
        fig, ax = plt.subplots(N, 1, figsize=(row_size * 1, col_size * N))
        for i, name in enumerate(names):
            if N > 1:
                ax_ = ax[i]
            else:
                ax_ = ax
            y = self.array(name)
            if len(y.shape) > 1 and y.shape[1] > 1:
                for j in range(y.shape[1]):
                    ax_.plot(t, y[:, j], label=f"{j}")
            else:
                ax_.plot(t, y, label="0")
            ax_.set_title(name)
            ax_.grid(True)
            ax_.legend()
        if N > 1:
            ax_ = ax[N - 1]
        else:
            ax_ = ax
        ax_.set_xlabel("time")
        ax_.legend()
        fig.tight_layout()
        plt.show()


class ArrayLogger:
    def __init__(self, N, specs):
        self.data = {name: np.zeros((N, *shape)) for name, shape in specs.items()}
        self.k = 0

    def log(self, **signals):
        for name, value in signals.items():
            self.data[name][self.k] = value
        self.k += 1

    def get(self, name):
        return self.data[name][: self.k]


class logger_context:
    def __init__(self, logger: SimLogger):
        self.logger = logger
        self.token = None

    def __enter__(self):
        self.token = _current_logger.set(self.logger)  # type: ignore
        return self.logger

    def __exit__(self, exc_type, exc, tb):
        _current_logger.reset(self.token)  # type: ignore


def get_logger():
    return _current_logger.get()


def record(**signals):
    logger = get_logger()
    if logger is not None:
        logger.record(**signals)


def next_step():
    logger = get_logger()
    if logger is not None:
        logger.next_step()
