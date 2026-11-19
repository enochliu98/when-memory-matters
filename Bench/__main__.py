"""Allow `python -m Bench` to run the bench runner."""
from Bench.runner import main
import sys

if __name__ == "__main__":
    sys.exit(main())
