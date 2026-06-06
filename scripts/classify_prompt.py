import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.modeling import classify_with_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    result = classify_with_model(args.prompt)

    print(result)


if __name__ == "__main__":
    main()
