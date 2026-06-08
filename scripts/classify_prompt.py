import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import resolve_model_path
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
    parser.add_argument(
        "--model-path", default=None,
        help="Model artifact to use. Defaults to model_path in service_config.json "
             "(same selection as the API), then the TF-IDF baseline; a missing file "
             "falls back to rule-based classification.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else resolve_model_path()

    logger.info("Classifying prompt (%d chars) from CLI using %s", len(args.prompt), model_path)
    result = classify_with_model(
        args.prompt, model_path=model_path, total_input_tokens=args.total_input_tokens,
    )

    print(result)


if __name__ == "__main__":
    main()
