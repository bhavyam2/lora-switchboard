class BatchContext:
    """
    Signals to LoRALinear layers that a heterogeneous batch is in flight.

    Holds a position map: adapter_id → list of batch indices that should
    receive that adapter's delta. Set by HeterogeneousBatchRunner before
    model.generate() and cleared immediately after.

    Safe as a class variable because the engine uses a single GPU thread
    (ThreadPoolExecutor max_workers=1), so only one forward pass runs at a time.
    """

    _positions: dict[str, list[int]] | None = None

    @classmethod
    def set(cls, positions: dict[str, list[int]]) -> None:
        cls._positions = positions

    @classmethod
    def get(cls) -> dict[str, list[int]] | None:
        return cls._positions

    @classmethod
    def clear(cls) -> None:
        cls._positions = None
