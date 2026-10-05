def division(a: float, b: float) -> dict:
    """Divide a by b."""

    if b == 0:
        raise ValueError("division by zero")

    return {"result": a / b}