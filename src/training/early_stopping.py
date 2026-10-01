"""Early stopping on a monitored validation value."""


class EarlyStopping:
    """Tracks the best value and signals a stop after `patience` epochs
    without improvement. mode="min" for a loss, "max" for a score."""

    def __init__(self, patience: int = 3, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record one epoch's value; True if it is a new best. NaN never is."""
        improved = (value < self.best - self.min_delta if self.mode == "min"
                    else value > self.best + self.min_delta)
        if improved:
            self.best, self.num_bad_epochs = value, 0
            return True
        self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return False

    def state_dict(self) -> dict:
        return {"best": self.best, "num_bad_epochs": self.num_bad_epochs,
                "should_stop": self.should_stop}

    def load_state_dict(self, state: dict) -> None:
        self.best = state.get("best", self.best)
        self.num_bad_epochs = state.get("num_bad_epochs", 0)
        self.should_stop = state.get("should_stop", False)
