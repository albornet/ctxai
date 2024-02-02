import sys
import logging
from app import app


HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984


def main():
    # Set logging level and format
    logger = logging.getLogger("cluster")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Run the app
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


if __name__ == "__main__":
    main()
    