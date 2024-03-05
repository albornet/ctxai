# Config
import os
import sys
try:
    import config
except:
    from . import config
import logging
logger = logging.getLogger("CTxAI")

# Utils
import csv
import json
import torchdata.datapipes.iter as dpi
from tqdm import tqdm
from typing import Iterator
from torchdata.dataloader2 import (
    DataLoader2,
    InProcessReadingService,
    MultiProcessingReadingService,
)
try:
    from cluster_utils import set_seeds
    from preprocess_utils import (
        ClinicalTrialFilter,
        CriteriaParser,
        CriteriaCSVWriter,
        CustomXLSXLineReader,
    )
except:
    from .cluster_utils import set_seeds
    from .preprocess_utils import (
        ClinicalTrialFilter,
        CriteriaParser,
        CriteriaCSVWriter,
        CustomXLSXLineReader,
    )

def main():
    """ Main script (if not run from a web-service)
    """
    # Logging
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Get current configuration and raw data path
    cfg = config.get_config()
    raw_data_path = os.path.join(
        cfg["BASE_DATA_DIR"],
        cfg["RAW_DATA_SUBDIR"],
        cfg["RAW_INPUT_FORMAT"],
    )
    
    # Parse criteria (main function not implemented yet)
    parse_data_fn(raw_data_path=raw_data_path)


def parse_data_fn(raw_data_path: str) -> None:
    """ Parse all CT files into lists of inclusion and exclusion criteria
    """
    # Get current configuration
    cfg = config.get_config()
    
    # Ensure reproducibility (needed here?)
    set_seeds(cfg["RANDOM_STATE"])
    
    # Load parsed data from previous run
    if cfg["LOAD_PREPROCESSED_DATA"]:
        logging.info(" - Eligibility criteria already parsed, skipping this step")
    
    # Parse data using torchdata logic
    else:
        logging.info(" - Building criteria from raw clinical trial texts")
        
        # Initialize output file with data headers
        preprocessed_dir = os.path.join(
            cfg["BASE_DATA_DIR"], cfg["POSTPROCESSED_SUBDIR"], cfg["RAW_INPUT_FORMAT"])
        os.makedirs(preprocessed_dir, exist_ok=True)
        csv_path = os.path.join(preprocessed_dir, "parsed_criteria.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cfg["PREPROCESSED_DATA_HEADERS"])
            
        # Initialize data processors
        too_many_workers = max(os.cpu_count() - 4, os.cpu_count() // 4)
        num_parse_workers = min(cfg["NUM_PARSE_WORKERS"], too_many_workers)
        ds = get_dataset(raw_data_path, cfg["RAW_INPUT_FORMAT"])
        if num_parse_workers == 0:
            rs = InProcessReadingService()
        else:
            rs = MultiProcessingReadingService(num_workers=num_parse_workers)
        dl = DataLoader2(ds, reading_service=rs)
        
        # Write parsed criteria to the output file
        for data in tqdm(dl, desc="Clinical trials processed so far", leave=False):
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(data)
                
        # Close data pipeline
        dl.shutdown()
        
    
def get_dataset(data_path: str, input_format: str) -> dpi.IterDataPipe:
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    # Load correct files from a directory
    if os.path.isdir(data_path):
        masks = "*.%s" % ("json" if input_format == "json" else "xlsx")
        files = dpi.FileLister(data_path, recursive=True, masks=masks)
        
    # Load correct file from a file path
    elif os.path.isfile(data_path):
        if not data_path.endswith(("json", "xlsx")):
            raise ValueError("File format not supported")
        files = dpi.IterableWrapper([data_path])
    
    # Handle exception
    else:
        raise FileNotFoundError(f"{data_path} is neither a file nor a directory")
    
    # Load data inside each file
    if input_format == "json":
        jsons = dpi.FileOpener(files, encoding="utf-8")
        dicts = CustomJsonParser(jsons)
        raw_samples = ClinicalTrialFilter(dicts)
    elif input_format == "ctxai":
        raw_samples = CustomXLSXLineReader(files)
    else:
        raise ValueError("Wrong input format selected.")
        
    # Parse criteria
    sharded_samples = dpi.ShardingFilter(raw_samples)
    parsed_samples = CriteriaParser(sharded_samples)
    written = CriteriaCSVWriter(parsed_samples)
    return written


class CustomJsonParser(dpi.JsonParser):
    """ Modificaion of dpi.JsonParser that handles empty files without error
    """
    def __iter__(self) -> Iterator[tuple[str, dict]]:
        for file_name, stream in self.source_datapipe:
            try:
                data = stream.read()
                stream.close()
                yield file_name, json.loads(data, **self.kwargs)
            except json.decoder.JSONDecodeError:
                print("Empty json file - skipping to next file.")
            

if __name__ == "__main__":
    main()
    