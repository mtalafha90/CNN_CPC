"""Entry point: predict the competition test set with a wide-encoder checkpoint."""
from .evaluate import predict as main

if __name__ == "__main__":
    main()
