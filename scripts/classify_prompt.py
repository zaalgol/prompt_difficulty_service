import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logging_config import get_logger
from app.modeling import classify_with_model

logger = get_logger("scripts.classify_prompt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--total-input-tokens", type=int, default=None,
        help="Prompt input-token count, used as an auxiliary feature by models trained with it",
    )
    args = parser.parse_args()

    logger.info("Classifying prompt (%d chars) from CLI", len(args.prompt))
    result = classify_with_model(args.prompt, total_input_tokens=args.total_input_tokens)

    print(result)


if __name__ == "__main__":
    main()
