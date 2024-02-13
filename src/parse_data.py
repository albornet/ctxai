import os
import csv
import logging
import torchdata.datapipes.iter as dpi
from torchdata.dataloader2 import (
    DataLoader2,
    InProcessReadingService,
    MultiProcessingReadingService,
)
from src.parse_utils import (
    ClinicalTrialFilter,
    CriteriaParser,
    CriteriaCSVWriter,
    CustomXLSXLineReader,
)
from tqdm import tqdm
try:
    from . import config as cfg
    from .cluster_utils import set_seeds
except:
    import src.config as cfg
    from cluster_utils import set_seeds
    
    
def main():
    """ Main script (if not run from a web-service)
    """
    # Format logging messages
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname).1s %(asctime)s] %(message)s",
    )
    
    # Parse criteria (main function not implemented yet)
    raise NotImplementedError


def parse_data_fn(raw_data_path: str) -> None:
    """ Parse all CT files into lists of inclusion and exclusion criteria
    """
    # Ensure reproducibility
    set_seeds(cfg.RANDOM_STATE)
    
    # Load parsed data from previous run
    if cfg.LOAD_PREPROCESSED_DATA:
        logging.info(" - Eligibility criteria already parsed, skipping this step")
    
    # Parse data using torchdata logic
    else:
        logging.info(" - Building criteria from raw clinical trial texts")
            
        # Initialize output file with data headers
        os.makedirs(cfg.PREPROCESSED_DIR, exist_ok=True)
        csv_path = os.path.join(cfg.PREPROCESSED_DIR, "parsed_criteria.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cfg.PREPROCESSED_DATA_HEADERS)
            
        # Initialize data processors
        ds = get_dataset(raw_data_path, cfg.RAW_INPUT_FORMAT)
        if cfg.NUM_PARSE_WORKERS == 0:
            rs = InProcessReadingService()
        else:
            rs = MultiProcessingReadingService(num_workers=cfg.NUM_PARSE_WORKERS)
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
        dicts = dpi.JsonParser(jsons)
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


if __name__ == "__main__":
    main()
    