import os
import json
import logging
from dataclasses import asdict
from flask import Flask, request, jsonify
from src import cluster_data_fn, ClusterOutput, cluster_config as cfg

app = Flask(__name__)

HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984
PATH_PREFIX = "/ct-risk/cluster"
RULE_PATH = "%s/predict" % PATH_PREFIX


@app.route(rule=RULE_PATH, methods=["POST"])
def predict():
    """ Cluster data found in data_dir found in request's field
    """
    # Validate required fields in JSON payload and send bad request otherwise
    request_data = request.get_json(force=True)
    requested_keys = ["user_id", "project_id", "input_dir"]
    if not all([k in request_data for k in requested_keys]):
        error_dict = {"error": "Missing field in request data"}
        return jsonify(error_dict), 400
    if "cluster_summarization_params" not in request_data:
        request_data["cluster_summarization_params"] = \
            cfg.DEFAULT_CLUSTER_SUMMARIZATION_PARAMS
    
    # Call process_instance function to prepare input tensors
    logging.info("Clustering procedure started")
    cluster_output = cluster_data_fn(
        model_type=cfg.APP_MODEL,
        input_dir=request_data["input_dir"],
        cluster_summarization_params=request_data["cluster_summarization_params"]
    )
    
    # Build unique output json file name
    unique_id = "%s_%s" % (request_data["project_id"], request_data["user_id"])
    file_name = os.path.join(cfg.RESULT_DIR, "%s_ec_clustering.csv" % unique_id)
    
    # Write output results to a file and return file path
    logging.info("Writing results to %s" % file_name)
    write_json_file(cluster_output, file_name)
    
    # Return jsonified file path corresponding to the written file
    return jsonify({"cluster_result_file_path": file_name})


def write_json_file(
    cluster_output: ClusterOutput,
    json_file_name: str,
) -> str:
    """ Convert cluster output to a dictionary and write it to a json file
    """    
    cluster_output_dict = asdict(cluster_output)
    json_data = json.dumps(cluster_output_dict, indent=4)
    with open(json_file_name, 'w') as file:
        file.write(json_data)
    
    
if __name__ == "__main__":
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)
    