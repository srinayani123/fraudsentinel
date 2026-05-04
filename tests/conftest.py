# Ensures pytest finds the src package when running from project root
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
