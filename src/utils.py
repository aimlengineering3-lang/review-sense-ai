"""
=========================================================
Review Sense AI
Utility Functions
=========================================================
"""

from pathlib import Path


def create_directory(path):
    """
    Create directory if it does not exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def separator(length=60):
    """
    Print separator line.
    """

    print("=" * length)