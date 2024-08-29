from transformers import AutoModel, AutoTokenizer


MODEL_STRS = [
    "pritamdeka/S-PubMedBert-MS-MARCO",
]


def download_model_resources(model_str):
    """ Import model and associated tokenizer to pre-download resources in docker
        cache folder
    """
    _ = AutoModel.from_pretrained(model_str)
    _ = AutoTokenizer.from_pretrained(model_str)


if __name__ == "__main__":
    for model_str in MODEL_STRS:
        download_model_resources(model_str)