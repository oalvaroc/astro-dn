"""AstroDN CLI entry point."""

import argparse


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tool to train and evaluate astronomical image denoising CNN models."  # noqa: E501
    )

    args = parser.parse_args()


main()
