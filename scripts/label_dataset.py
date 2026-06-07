import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dataset import label_dataset
from app.logging_config import get_logger

logger = get_logger("scripts.label_dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original report.json")
    parser.add_argument("--output", required=True, help="Path for labeled output JSON")
    args = parser.parse_args()

    logger.info("Labeling dataset from CLI: %s -> %s", args.input, args.output)
    total, counts = label_dataset(args.input, args.output)

    print(f"Labeled {total} prompts")
    print(f"Label counts: {counts}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
