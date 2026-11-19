"""Optional live API check. Importing this module makes no requests."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("volcengine", "deepseek"), required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    from langpsro.utils import make_llm_client

    client = make_llm_client(args.provider, model=args.model)
    response = client("Reply with OK.", max_tokens=32)
    if not response:
        raise SystemExit("The provider returned no text.")
    print("The provider returned text successfully.")


if __name__ == "__main__":
    main()
