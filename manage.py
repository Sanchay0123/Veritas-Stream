#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import signal # <--- Added

def main():
    """Run administrative tasks."""
    # KEEPING YOUR CORRECT SETTINGS PATH:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

    # --- START FIX: Handle Ctrl+C (SIGINT) ---
    def signal_handler(sig, frame):
        print("\n👋 Gracefully closing Django Server & freeing port...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    # --- END FIX ---

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()