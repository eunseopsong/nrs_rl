"""Finite forward-path episode completion and the 336-second timeout."""

def episode_ends(cursor, length, elapsed):
    completed = cursor >= length
    timeout = elapsed >= 336.
    return completed, completed | timeout
