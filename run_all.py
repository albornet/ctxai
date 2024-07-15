from src import config
import json
import subprocess
import time
import os
import signal

config.load_default_config()
logger = config.CTxAILogger("INFO")


def run_wsgi_server():
    """ Function to start the WSGI server
    """
    logger.info("Starting WSGI server")
    command = ["python", "wsgi.py"]
    return subprocess.Popen(command)


def run_curl_command(
    chosen_cond_ids: list[str],
    chosen_cond_lvl: int,
    chosen_itrv_lvl: int,
):
    """ Query the clustering API with one set of experimental conditions

    Args:
        chosen_cond_ids (list[str]): filter of conditions used by CTs
        chosen_cond_lvl (int): condition level from which labels are built
        chosen_itrv_lvl (int): intervention level from which labels are built

    Returns:
        response from the clustering API
    """
    query_dict = {
        "SCRIPT_MODE": True,
        "ENVIRONMENT": "ctgov",
        "CHOSEN_COND_IDS": chosen_cond_ids,
        "CHOSEN_COND_LVL": chosen_cond_lvl,
        "CHOSEN_ITRV_LVL": chosen_itrv_lvl,
    }
    query_json = json.dumps(query_dict)
    
    command = [
        "curl", "-X", "POST", "http://0.0.0.0:8984/ct-risk/cluster/predict",
        "-H", "Content-Type: application/json",
        "-d", query_json,
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result


def main():
    """ Run the clustering pipeline for many different experimental conditions
    """
    # Start the WSGI server
    wsgi_process = run_wsgi_server()
    
    # Give the server some time to start
    time.sleep(20)

    try:
        # Define your sequence of environments and data paths
        tasks = [
            {"chosen_cond_ids": ["C01"], "chosen_cond_lvl": 2, "chosen_itrv_lvl": 1},
            {"chosen_cond_ids": ["C01"], "chosen_cond_lvl": 3, "chosen_itrv_lvl": 2},
            {"chosen_cond_ids": ["C01"], "chosen_cond_lvl": 4, "chosen_itrv_lvl": 3},
            {"chosen_cond_ids": ["C04"], "chosen_cond_lvl": 2, "chosen_itrv_lvl": 1},
            {"chosen_cond_ids": ["C04"], "chosen_cond_lvl": 3, "chosen_itrv_lvl": 2},
            {"chosen_cond_ids": ["C04"], "chosen_cond_lvl": 4, "chosen_itrv_lvl": 3},
            {"chosen_cond_ids": ["C14"], "chosen_cond_lvl": 2, "chosen_itrv_lvl": 1},
            {"chosen_cond_ids": ["C14"], "chosen_cond_lvl": 3, "chosen_itrv_lvl": 2},
            {"chosen_cond_ids": ["C14"], "chosen_cond_lvl": 4, "chosen_itrv_lvl": 3},
            {"chosen_cond_ids": ["C20"], "chosen_cond_lvl": 2, "chosen_itrv_lvl": 1},
            {"chosen_cond_ids": ["C20"], "chosen_cond_lvl": 3, "chosen_itrv_lvl": 2},
            {"chosen_cond_ids": ["C20"], "chosen_cond_lvl": 4, "chosen_itrv_lvl": 3},
            {"chosen_cond_ids": ["C01", "C04", "C14", "C20"], "chosen_cond_lvl": 2, "chosen_itrv_lvl": 1},
            {"chosen_cond_ids": ["C01", "C04", "C14", "C20"], "chosen_cond_lvl": 3, "chosen_itrv_lvl": 2},
            {"chosen_cond_ids": ["C01", "C04", "C14", "C20"], "chosen_cond_lvl": 4, "chosen_itrv_lvl": 3},
        ]

        for task in tasks:
            logger.info(f"Sending curl request with {task}")
            result = run_curl_command(**task)
            if result.returncode == 0:
                logger.info("Curl request sent and received successfully")
            else:
                logger.error(f"Curl request failed (code {result.returncode})")
                logger.error(f"Detailed error: {result.stderr}")
            time.sleep(10)  # delay to avoid overloading the server
    
    finally:
        # Terminate the WSGI server process
        os.kill(wsgi_process.pid, signal.SIGTERM)
        

if __name__ == "__main__":
    main()
