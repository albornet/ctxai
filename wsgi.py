import logging
from app import app


def main():
    # Set logging level and format
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname).1s %(asctime)s] %(message)s",
    )
    
    # Run the app
    app.run()


if __name__ == "__main__":
    main()
    