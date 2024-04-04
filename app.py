import logging
from src import config, parse_data_fn, cluster_data_fn
from flask import Flask, jsonify, request
config.load_default_config()
logger = logging.getLogger("cluster")

app = Flask(__name__)

PATH_PREFIX = "/ct-risk/cluster"
RULE_PATH = "%s/predict" % PATH_PREFIX


@app.route(rule=RULE_PATH, methods=["POST"])
def predict():
    """ Cluster data found in data_dir found in request's field
    """
    # Initialization and validation of required fields in JSON payload
    logger.info("Starting eligibility criteria clustering pipeline")
    request_data = request.get_json(force=True)
    required_keys = [
        "ENVIRONMENT",
        "DATA_PATH",
        "USER_ID",
        "PROJECT_ID",
        "USER_FILTERING",
        "EMBEDDING_MODEL_ID",
    ]
    if not all([k in request_data for k in required_keys]):
        error_dict = {"error": "Missing field in request data"}
        return jsonify(error_dict), 400
    
    # Update in-memory configuration using request data
    config.update_config(request_data)
    
    # Parse raw data into pre-processed data files
    logger.info("Parsing criterion texts into individual criteria")
    parse_data_fn()
    
    # Cluster pre-processed data
    logger.info("Clustering procedure started")
    cluster_output = cluster_data_fn(request_data["EMBEDDING_MODEL_ID"])
    
    # Return jsonified file paths corresponding to the written data and plot
    logger.info("Clustering procedure finished")
    return jsonify({
        "cluster_json_path": cluster_output.json_path,
        "cluster_visualization_paths": cluster_output.visualization_paths,
        "cluster_raw_ec_list_path": cluster_output.raw_ec_list_path,
    })

    
if __name__ == "__main__":
    host_address = '0.0.0.0'
    port_number = 8984
    app.run(debug=False, host=host_address, port=port_number)
    