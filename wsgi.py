import logging
from app import app

HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984
    

def main():
    # Set logging level and format
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname).1s %(asctime)s] %(message)s",
    )
    
    # Run the app
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


if __name__ == "__main__":
    main()
    