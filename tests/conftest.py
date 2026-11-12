import os
import sys

# config.py builds the Gemini client at import time and refuses to start without
# a key. A dummy value lets the tests import it; nothing here calls the API.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
