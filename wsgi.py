import logging
from app import app


HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984


def main():
    set_logger()   
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


def set_logger(name="cluster"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = 0  # prevent logging from propagating to the root logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    

if __name__ == "__main__":
    main()
