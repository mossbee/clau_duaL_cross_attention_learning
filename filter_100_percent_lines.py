import argparse
from pathlib import Path


def filter_lines_with_100_percent(input_path: Path, output_path: Path) -> int:
    """Read input file and write only lines containing '100%' to output.

    Returns the number of lines written.
    """
    lines_written = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if "100%" in line or "Current" in line or "Evaluating val: 100%" in line or "/100" in line or "Train Loss" in line or "Val Metric" in line or "New best model saved with metric" in line:
                outfile.write(line)
                lines_written += 1
    return lines_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter lines containing '100%' from train_log.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_log.txt"),
        help="Path to input log file (default: train_log.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_log_100.txt"),
        help="Path to output file with only lines containing '100%' (default: train_log_100.txt)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    written = filter_lines_with_100_percent(args.input, args.output)
    print(f"Wrote {written} lines containing '100%' to {args.output}")


if __name__ == "__main__":
    main()


