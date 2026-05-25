"""
context_analyzer.py

Figures out what kind of workflow we're looking at and which steps
are handled by humans vs. automated systems.
"""

import pandas as pd


def detect_workflow_context(df: pd.DataFrame) -> dict:
    """Classify the workflow type and split tasks into human vs system.

    Uses keyword matching on task names — not perfect but covers the
    most common workflow patterns we've seen.
    """
    if df.empty or "task" not in df.columns or "user" not in df.columns:
        return {
            "workflow_type": "Unknown",
            "human_tasks": [],
            "system_tasks": []
        }

    unique_tasks = df["task"].unique()
    task_str = " ".join(unique_tasks).lower()

    # keyword-based workflow classification — add more patterns as needed
    if any(k in task_str for k in ["lead", "proposal", "deal"]):
        w_type = "Sales Workflow"
    elif any(k in task_str for k in ["ticket", "support", "incident"]):
        w_type = "Customer Support Workflow"
    elif any(k in task_str for k in ["onboarding", "hr", "hire"]):
        w_type = "Employee Onboarding Workflow"
    else:
        w_type = "Generic Business Workflow"

    # split tasks by whether they're always run by "system" user
    human_tasks = []
    system_tasks = []

    for task in unique_tasks:
        users = df.loc[df["task"] == task, "user"].unique()
        if len(users) == 1 and users[0].lower() == "system":
            system_tasks.append(task)
        else:
            human_tasks.append(task)

    return {
        "workflow_type": w_type,
        "human_tasks": sorted(human_tasks),
        "system_tasks": sorted(system_tasks)
    }
