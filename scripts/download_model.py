import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

load_dotenv()

from huggingface_hub import snapshot_download

MAX_RETRIES = 5
RETRY_DELAY = 5


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    full_name = f"sentence-transformers/{model_name}"
    print(f"Downloading embedding model: {full_name}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = snapshot_download(full_name)
            print(f"Model downloaded and cached at: {path}")
            return
        except Exception as exc:
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                print(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"All {MAX_RETRIES} attempts failed. Check your internet connection and retry.")
                sys.exit(1)


if __name__ == "__main__":
    main()
