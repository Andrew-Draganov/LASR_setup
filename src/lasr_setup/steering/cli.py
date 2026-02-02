"""CLI for running steering vector extraction experiments."""

import argparse

from lasr_setup.steering.extract import extract_multiple_steering_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering vectors from a language model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/caps_dataset.jsonl",
        help="Path to JSONL dataset (default: datasets/caps_dataset.jsonl)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Which layer to extract activations from",
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1,
        help="Number of orthogonal steering vectors to extract (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scratch/steering_vectors",
        help="Directory to save steering vectors (default: scratch/steering_vectors)",
    )

    args = parser.parse_args()

    extract_multiple_steering_vectors(
        model_name="Qwen/Qwen3-0.6B",
        dataset_path=args.dataset,
        layer=args.layer,
        num_vectors=args.num_vectors,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
