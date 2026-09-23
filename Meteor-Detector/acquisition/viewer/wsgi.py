"""Gunicorn entry point for the private Pi viewer.

Set RRO_DATA_DIR in the service environment before importing this module.
"""

from server import app

