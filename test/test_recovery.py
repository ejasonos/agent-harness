from agent.recovery import RecoveryManager


recovery = RecoveryManager(max_retries_per_tool=2)

print(recovery.evaluate(
    "division_tool",
    "Division by zero",
))

print(recovery.evaluate(
    "division_tool",
    "Division by zero",
))

print(recovery.evaluate(
    "division_tool",
    "Division by zero",
))

recovery.reset("division_tool")

print(recovery.evaluate(
    "division_tool",
    "Division by zero",
))