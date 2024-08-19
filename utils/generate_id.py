import random
import string

import argparse


def generate_random_id(length):
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()

    random_id = generate_random_id(args.length)

    print(f"random id generated: {random_id}")
