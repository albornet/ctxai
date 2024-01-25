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
except:
    import src.config as cfg
    
    
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


def parse_data_fn() -> None:
    """ Parse all CT files into lists of inclusion and exclusion criteria
    """
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
        ds = get_dataset(cfg.RAW_DATA_DIR, cfg.RAW_INPUT_FORMAT)
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
        
    
def get_dataset(data_dir, input_format, shuffle=False):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    # Load correct files
    masks = "*.%s" % ("json" if input_format == "json" else "xlsx")
    files = dpi.FileLister(data_dir, recursive=True, masks=masks)
    sharded = dpi.ShardingFilter(files)  # remove this if single worker?
    if shuffle: sharded = dpi.Shuffler(sharded)
    
    # Load data inside each file
    if input_format == "json":
        jsons = dpi.FileOpener(sharded, encoding="utf-8")
        dicts = dpi.JsonParser(jsons)
        raw_samples = ClinicalTrialFilter(dicts)
    elif input_format == "ctxai":
        raw_samples = CustomXLSXLineReader(sharded)
    else:
        raise ValueError("Wrong input format selected.")
        
    # Parse criteria
    parsed_samples = CriteriaParser(raw_samples)
    written = CriteriaCSVWriter(parsed_samples)
    return written


if __name__ == "__main__":
    main()
    