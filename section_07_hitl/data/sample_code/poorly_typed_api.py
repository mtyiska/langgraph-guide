# poorly_typed_api.py — missing type hints, poor error handling

def get_user(user_id):
    # Missing return type, no validation of user_id
    if user_id <= 0:
        return None
    # Simulated DB fetch
    return {"id": user_id, "name": "Alice", "email": "alice@example.com"}


def create_order(user_id, items, discount=0):
    # No type hints, no validation that items is non-empty
    # discount could be negative, no guard
    total = sum(item["price"] * item["quantity"] for item in items)
    total = total * (1 - discount)
    return {"user_id": user_id, "items": items, "total": total}


def update_inventory(product_id, quantity_delta):
    # No check that quantity_delta won't push stock negative
    # Missing type annotations throughout
    current_stock = 100  # simulated
    new_stock = current_stock + quantity_delta
    return new_stock


def send_email(to, subject, body, cc=None, bcc=None):
    # No validation that `to` is a valid email
    # No return type annotation
    # cc and bcc not validated
    print(f"Sending to {to}: {subject}")
    return True


class UserCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        # KeyError if key not present — should use .get()
        return self.cache[key]

    def set(self, key, value, ttl=None):
        # ttl parameter accepted but silently ignored
        self.cache[key] = value

    def invalidate(self, key):
        # No check if key exists before deleting
        del self.cache[key]