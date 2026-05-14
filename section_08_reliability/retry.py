import time
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    jitter: bool = True,
):
    """
    Decorator that retries a function with exponential backoff on failure.

    Args:
        max_attempts: Total number of attempts before giving up.
        base_delay:   Delay in seconds before the second attempt.
                      Doubles on each subsequent attempt.
        max_delay:    Cap on the delay regardless of attempt count.
        exceptions:   Tuple of exception types to retry on.
                      All other exceptions propagate immediately.
        jitter:       If True, adds random noise to delays to prevent
                      thundering herd when many calls fail simultaneously.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.warning(
                            f"{func.__name__}: all {max_attempts} attempts failed. "
                            f"Last error: {e}"
                        )
                        break

                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)

                    logger.info(
                        f"{func.__name__}: attempt {attempt} failed ({e}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception — propagate immediately
                    raise

            raise last_exception

        return wrapper
    return decorator


class RetryStats:
    """Tracks retry statistics across tool calls for monitoring."""

    def __init__(self):
        self.attempts: dict[str, list[int]] = {}   # tool_name -> list of attempt counts

    def record(self, tool_name: str, attempts_used: int):
        self.attempts.setdefault(tool_name, []).append(attempts_used)

    def summary(self) -> dict:
        return {
            name: {
                "calls": len(counts),
                "total_attempts": sum(counts),
                "retry_rate": sum(1 for c in counts if c > 1) / len(counts),
                "avg_attempts": sum(counts) / len(counts),
            }
            for name, counts in self.attempts.items()
        }


# Global stats tracker — import and use in tools
retry_stats = RetryStats()