"""Early stopping on a monitored validation metric."""


class EarlyStopping:
    """Tracks the best value of a monitored metric and signals when to stop.

    mode="min" for a loss (lower is better), mode="max" for a score like F1.
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Update state with the latest epoch's value. Returns True if it's a new best."""
        if self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False
