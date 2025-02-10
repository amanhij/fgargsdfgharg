import asyncio
from threading import Lock


class SharedState:
    """Manages the shared application state across modules."""

    def __init__(self):
        # Initialize the shared state
        self._state = {
            "tracked_tokens": {},
            "market_cap_history": {},
            "tg_popularity": {},
            "velocities": {},
            "authenticated": False,
            "temp_client": None,
            "interval_market_cap_history": {},  # {symbol: [(timestamp, market_cap)]}
            "dexscreener_data": {},
            "dexscreener_changes": {},
            "liquidity":{},
            "stagnant_tokens": {},
            "filtered_tokens": {},
            "active_positions": {},
            "processed_tokens": {},
            "slippage":{},
            "priority_fee":{},
        }
        self._lock = Lock()  # Ensures thread-safe access

    def get(self, key, default=None):
        """Thread-safe getter for a state key."""
        with self._lock:
            return self._state.get(key)

    def set(self, key, value):
        """Thread-safe setter for a state key."""
        with self._lock:
            self._state[key] = value

    def update(self, key, update_func):
        """Thread-safe update of a state key using a function."""
        with self._lock:
            if key in self._state:
                self._state[key] = update_func(self._._state[key])
            else:
                raise KeyError(f"Key '{key}' does not exist in state.")

    def get_full_state(self):
        """Returns the entire state (useful for debugging or exporting)."""
        with self._lock:
            return self._state.copy()

    def clear(self):
        """Resets the entire shared state."""
        with self._lock:
            self._state = {}
