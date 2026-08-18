"""Entry point: score a wide-encoder checkpoint on the expert studies."""
from .evaluate import validate as main

if __name__ == "__main__":
    main()
