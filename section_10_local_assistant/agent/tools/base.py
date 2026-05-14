import time
import random
import logging
from functools import wraps
from typing import Callable, Literal, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay:   float = 1.0,
    max_delay:    float = 30.0,
    exceptions:   tuple = (Exception,),
    jitter:       bool  = True,
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)
                    logger.info(f"{func.__name__}: attempt {attempt} failed. Retry in {delay:.2f}s")
                    time.sleep(delay)
                except Exception:
                    raise
            raise last_exc
        return wrapper
    return decorator


class ToolSuccess(BaseModel):
    status: Literal["success"] = "success"
    result: str
    metadata: dict = {}


class ToolError(BaseModel):
    status:     Literal["error"] = "error"
    error_type: Literal["not_found", "permission_denied", "timeout", "invalid_input", "unknown"]
    message:    str
    recoverable: bool
    suggestion:  str


ToolResponse = Union[ToolSuccess, ToolError]