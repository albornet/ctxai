import logging
from flask import Flask, jsonify, request
from src import config as cfg, parse_data_fn, cluster_data_fn

app = Flask(__name__)

PATH_PREFIX = "/ct-risk/cluster"
RULE_PATH = "%s/predict" % PATH_PREFIX


@app.route(rule=RULE_PATH, methods=["POST"])
def predict():
    """ Cluster data found in data_dir found in request's field
    """
    # Validate required fields in JSON payload and send bad request otherwise
    request_data = request.get_json(force=True)
    requested_keys = ["user_id", "project_id", "raw_data_path"]
    if not all([k in request_data for k in requested_keys]):
        error_dict = {"error": "Missing field in request data"}
        return jsonify(error_dict), 400
    if "cluster_summarization_params" not in request_data:
        request_data["cluster_summarization_params"] = \
            cfg.DEFAULT_CLUSTER_SUMMARIZATION_PARAMS
    if "model_id" not in request_data:
        request_data["model_id"] = cfg.DEFAULT_MODEL_ID
    
    # Parse raw data into pre-processed data files
    logging.info("Parsing criterion texts into individual criteria")
    parse_data_fn(raw_data_path=request_data["raw_data_path"])
    
    # Cluster pre-processed data
    logging.info("Clustering procedure started")
    cluster_output = cluster_data_fn(
        model_id=request_data["model_id"],
        user_id=request_data["user_id"],
        project_id=request_data["project_id"],
        cluster_summarization_params=request_data["cluster_summarization_params"]
    )
    
    # Return jsonified file paths corresponding to the written data and plot
    return jsonify({
        "cluster_json_path": cluster_output.json_path,
        "cluster_visualization_paths": cluster_output.visualization_paths,
    })

    
if __name__ == "__main__":
    host_address = '0.0.0.0'
    port_number = 8984
    app.run(debug=False, host=host_address, port=port_number)
    