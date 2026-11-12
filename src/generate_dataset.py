"""
generate_dataset.py

Builds two synthetic event logs for demos and tests:

* sample_sales_pipeline.csv  - a lead-to-deal sales process
* sample_support_tickets.csv - a customer support process

Both are seeded, so the output is identical on every run. They contain a slow
step, handoff queues, rework loops and skipped steps. The support sample also
includes a handful of deliberately bad rows (end time before start time) to
show the data-quality handling.

Usage:  python src/generate_dataset.py
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta

END_OF_WINDOW = datetime(2026, 3, 16, 12, 0, 0)
WINDOW_DAYS = 90

SCENARIOS: dict[str, dict] = {
    "sample_sales_pipeline": {
        "cases": 300,
        "steps": ["Lead Created", "Lead Reviewed", "Manager Approval", "Proposal Sent", "Deal Closed"],
        "users": {
            "Lead Created": ["alice", "bob", "charlie", "diana"],
            "Lead Reviewed": ["eve", "frank", "grace"],
            "Manager Approval": ["harry", "isabella"],
            "Proposal Sent": ["jack", "kate", "leo"],
            "Deal Closed": ["mike", "nancy"],
        },
        # (min, max) hours
        "duration": {
            "Lead Created": (0.5, 2.0), "Lead Reviewed": (1.0, 4.0),
            "Manager Approval": (1.0, 6.0), "Proposal Sent": (0.5, 3.0),
            "Deal Closed": (0.5, 2.0),
        },
        "gap": {
            "Lead Reviewed": (0.5, 4.0), "Manager Approval": (0.5, 3.0),
            "Proposal Sent": (0.5, 2.0), "Deal Closed": (0.5, 3.0),
        },
        "slow_step": ("Manager Approval", 0.35, (12.0, 48.0)),
        "rework": (("Lead Reviewed", "Manager Approval"), 0.10),
        "skippable": (["Lead Reviewed"], 0.05),
        "bad_rows": 0,
    },
    "sample_support_tickets": {
        "cases": 300,
        "steps": ["Ticket Opened", "Auto-Triage", "Agent Assigned", "Investigation",
                  "Customer Confirmation", "Resolved"],
        "users": {
            "Ticket Opened": ["portal", "email-gateway"],
            "Auto-Triage": ["system"],
            "Agent Assigned": ["dispatch", "lena", "omar"],
            "Investigation": ["lena", "omar", "priya", "sam"],
            "Customer Confirmation": ["lena", "omar", "priya", "sam"],
            "Resolved": ["lena", "omar", "priya", "sam"],
        },
        "duration": {
            "Ticket Opened": (0.05, 0.2), "Auto-Triage": (0.02, 0.1),
            "Agent Assigned": (0.1, 0.4), "Investigation": (0.5, 4.0),
            "Customer Confirmation": (0.2, 2.0), "Resolved": (0.1, 0.3),
        },
        # the biggest delay here is the queue before an agent picks the ticket up
        "gap": {
            "Auto-Triage": (0.02, 0.2), "Agent Assigned": (1.0, 10.0),
            "Investigation": (0.1, 1.0), "Customer Confirmation": (0.5, 6.0),
            "Resolved": (0.05, 0.5),
        },
        "slow_step": ("Investigation", 0.25, (6.0, 30.0)),
        "rework": (("Investigation", "Customer Confirmation"), 0.12),
        "skippable": (["Customer Confirmation"], 0.07),
        "bad_rows": 4,
    },
}


def _hours(rng: random.Random, bounds: tuple[float, float]) -> timedelta:
    return timedelta(hours=rng.uniform(*bounds))


def _plan_steps(rng: random.Random, spec: dict) -> list[str]:
    """Choose the path one case follows: standard, rework loop or skip."""
    steps = list(spec["steps"])
    roll = rng.random()
    rework_pair, rework_prob = spec["rework"]
    skippable, skip_prob = spec["skippable"]

    if roll < rework_prob:
        # the pair is done a second time, e.g. approval rejected -> review again
        last = steps.index(rework_pair[1])
        steps = steps[: last + 1] + list(rework_pair) + steps[last + 1:]
    elif roll < rework_prob + skip_prob:
        steps.remove(rng.choice(skippable))
    return steps


def generate_case(rng: random.Random, spec: dict, case_number: int) -> list[dict]:
    start = END_OF_WINDOW - timedelta(days=rng.uniform(0, WINDOW_DAYS))
    current = start.replace(hour=rng.randint(8, 17), minute=rng.randint(0, 59),
                            second=0, microsecond=0)
    slow_step, slow_prob, slow_extra = spec["slow_step"]
    events: list[dict] = []

    for i, step in enumerate(_plan_steps(rng, spec)):
        if i > 0 and step in spec["gap"]:
            current += _hours(rng, spec["gap"][step])
        duration = _hours(rng, spec["duration"][step])
        if step == slow_step and rng.random() < slow_prob:
            duration += _hours(rng, slow_extra)

        end = current + duration
        events.append({
            "case_id": f"CASE-{case_number:04d}",
            "task": step,
            "start_time": current.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "user": rng.choice(spec["users"][step]),
        })
        current = end
    return events


def generate_dataset(name: str, seed: int = 42) -> list[dict]:
    spec = SCENARIOS[name]
    rng = random.Random(f"{seed}-{name}")
    events: list[dict] = []
    for case_number in range(1, spec["cases"] + 1):
        events.extend(generate_case(rng, spec, case_number))

    # swap start/end on a few rows to simulate bad exports
    for index in rng.sample(range(len(events)), spec["bad_rows"]):
        events[index]["start_time"], events[index]["end_time"] = (
            events[index]["end_time"], events[index]["start_time"])
    return events


def save_to_csv(events: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "task", "start_time", "end_time", "user"])
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    for name in SCENARIOS:
        events = generate_dataset(name)
        path = os.path.join(root, "data", f"{name}.csv")
        save_to_csv(events, path)
        print(f"Wrote {path} ({len(events)} rows)")


if __name__ == "__main__":
    main()
