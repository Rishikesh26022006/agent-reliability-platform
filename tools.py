import random

FAKE_ORDERS = {
    "8891": {"item": "Wireless Mouse", "status": "delivered", "amount": 24.99, "delivery_date": "2026-07-18"},
    "4432": {"item": "Bluetooth Headphones", "status": "in_transit", "amount": 79.99, "delivery_date": "2026-07-25"},
    "1029": {"item": "Laptop Stand", "status": "delivered", "amount": 34.50, "delivery_date": "2026-07-15"},
}

POLICIES = {
    "damaged_item": "Full refund issued if reported within 30 days of delivery, no return required for items under $50.",
    "late_delivery": "No automatic refund for late delivery unless delayed more than 14 days past estimated date.",
    "wrong_item": "Full refund or replacement, customer does not need to return the wrong item.",
    "change_of_mind": "Refund only if item is unused and returned within 14 days; customer pays return shipping.",
}


def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID and return its details (item, status, amount, delivery date)."""
    order = FAKE_ORDERS.get(order_id)
    if order:
        return {"found": True, "order_id": order_id, **order}
    return {"found": False, "order_id": order_id, "error": "Order not found"}


def check_policy(category: str) -> dict:
    """Check the refund/return policy for a given category.
    Valid categories: damaged_item, late_delivery, wrong_item, change_of_mind."""
    policy = POLICIES.get(category)
    if policy:
        return {"found": True, "category": category, "policy": policy}
    return {"found": False, "category": category, "error": "Unknown policy category"}


def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund for a given order ID and amount. Only call this after confirming
    the order exists and checking that policy allows a refund."""
    if order_id not in FAKE_ORDERS:
        return {"success": False, "error": "Cannot refund - order not found"}
    return {"success": True, "order_id": order_id, "refunded_amount": amount}


def escalate_to_human(order_id: str, reason: str) -> dict:
    """Escalate this case to a human support agent. Use this for anything you're not
    confident about, fraud concerns, angry customers, or policy edge cases."""
    return {"escalated": True, "order_id": order_id, "reason": reason, "ticket_id": f"ESC-{random.randint(1000,9999)}"}


if __name__ == "__main__":
    print(lookup_order("8891"))
    print(check_policy("damaged_item"))
    print(issue_refund("8891", 24.99))
    print(escalate_to_human("8891", "Customer is disputing charge"))