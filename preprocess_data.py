import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torchdata.datapipes.iter as dpi
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from parsing_utils import (
    ClinicalTrialFilter,
    CriteriaParser,
    CriteriaCSVWriter,
    CustomXLSXLineReader,
    CustomDictLineReader,
)
from statistics import mean, stdev, median, mode
from tqdm import tqdm
from typing import Union


INPUT_FORMAT = 'dict'  # 'json', 'xlsx', 'dict'
LOAD_CSV_RESULTS = False
EXAMPLE = False
DATA_DIR = os.path.join('data', 'raw_files', INPUT_FORMAT)
RESULT_DIR = os.path.join('data', 'preprocessed', INPUT_FORMAT)
PLOT_PATH = os.path.join(RESULT_DIR, 'criteria_length_histogram.png')
CSV_PATH = os.path.join(RESULT_DIR, 'parsed_criteria.csv')
CSV_HEADERS = [
    'criteria paragraph', 'complexity', 'ct path', 'label', 'phases',
    'conditions', 'condition_ids', 'intervention_ids', 'category', 'context',
    'subcontext', 'individual criterion',
]
ENCODING = 'utf-8'
NUM_STEPS = 110000
NUM_WORKERS = 0  # 12
NUM_WORKERS = min(NUM_WORKERS, max(os.cpu_count() - 4, os.cpu_count() // 4))
PREFETCH_FACTOR = None if NUM_WORKERS == 0 else 2
if EXAMPLE:
    NUM_STEPS = NUM_WORKERS
    PLOT_PATH = os.path.join(RESULT_DIR, 'criteria_length_histogram_example.png')
    CSV_PATH = os.path.join(RESULT_DIR, 'parsed_criteria_example.csv')


def main():
    """ Parse all CT files into lists of inclusion and exclusion criteria.
    """
    if not LOAD_CSV_RESULTS:
        # Initialize output file with data headers
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            
        # Initialize data processors
        ds = get_dataset(DATA_DIR, INPUT_FORMAT, shuffle=True)
        rs = MultiProcessingReadingService(
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )
        dl = DataLoader2(ds, reading_service=rs)
        
        # Write parsed criteria to the output file
        for i, data in tqdm(enumerate(dl),
                            total=NUM_STEPS,
                            desc='Parsing criteria',
                            leave=False):
            with open(CSV_PATH, 'a', newline='', encoding=ENCODING) as f:
                writer = csv.writer(f)
                writer.writerows(data)
            if i >= NUM_STEPS - 1: break
        dl.shutdown()
    
    # Evaluate csv result file, and compute some statistics
    compute_result_stats(CSV_PATH, PLOT_PATH)
    
    
def get_dataset(data_dir, input_format, shuffle=False):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    # Load correct files
    masks = '*.%s' % ('xlsx' if input_format == 'dict' else input_format)
    files = dpi.FileLister(data_dir, recursive=True, masks=masks)
    sharded = dpi.ShardingFilter(files)  # split file processing by shards
    if shuffle: sharded = dpi.Shuffler(sharded)
    
    # Load data inside each file
    if input_format == 'json':
        jsons = dpi.FileOpener(sharded, encoding=ENCODING)
        dicts = dpi.JsonParser(jsons)
        raw_samples = ClinicalTrialFilter(dicts)
    elif input_format == 'xlsx':
        raw_samples = CustomXLSXLineReader(sharded)
    elif input_format == 'dict':
        raw_samples = CustomDictLineReader(sharded)
    else:
        raise ValueError('Wrong input format selected.')
        
    # Parse criteria (if not done already)
    if input_format == 'dict':
        parsed_samples = raw_samples  # already parsed in the raw file
    else:
        parsed_samples = CriteriaParser(raw_samples)
    written = CriteriaCSVWriter(parsed_samples)
    return written

    
def compute_result_stats(result_path: str,
                         plot_path: str,
                         is_long_criterion_thresh: int=400,
                         ) -> None:
    """ Evaluate parsing and how some interesting statistics
    """
    # Load data and compute some numbers
    df = pd.read_csv(result_path).dropna(subset='individual criterion')
    df.to_csv(result_path)  # 4 individual criterions were NaN -> remove them
    n_easy = len(df.loc[df['complexity'] == 'easy'])
    n_hard = len(df.loc[df['complexity'] == 'hard'])
    n_criteria = len(df)
    criteria_lens = [len(c) for c in df['individual criterion']]
    x_to_add = {'Long threshold': {'k': is_long_criterion_thresh}}
    misc_to_add = {'N easy CTs': n_easy,
                   'N hard CTs': n_hard,
                   'N criteria': n_criteria}
    
    # Plot results and statistics
    _, ax = plt.subplots()
    plot_histogram(criteria_lens, ax, x_to_add, misc_to_add)
    plt.xlim([-0.1 * is_long_criterion_thresh, 2 * is_long_criterion_thresh])
    plt.legend()
    plt.savefig(plot_path, dpi=150)


def plot_histogram(data: list[Union[int, float]],
                   ax: plt.Axes,
                   x_to_add: dict[str, dict[str, Union[int, float]]]={},
                   misc_to_add: dict[str, Union[int, float]]={},
                   line_p={'linestyle': 'dashed', 'linewidth': 2},
                   ) -> None:
    """ Function to plot a histogram and various statistics
    """
    # Add various additional information
    for label, m in misc_to_add.items():
        if isinstance(m, int):
            ax.plot([], [], ' ', label='%s: %i' % (label, m))
        else:
            ax.plot([], [], ' ', label='%s: %.2fi' % (label, m))
    
    # Add additional x values
    for label, x_dict in x_to_add.items():
        c, x = next(iter(x_dict.items()))
        if isinstance(x, int):
            ax.axvline(x, c=c, **line_p, label='%s: %i' % (label, x))
        else:
            ax.axvline(x, c=c, **line_p, label='%s: %.2f' % (label, x))
            
    # Plot vertical lines for mean, median, and mode
    mean_, stdev_ = mean(data), stdev(data)
    median_, mode_ = median(data), mode(data)
    bins = int((max(data) - min(data)) * 10 / stdev_)
    ax.hist(data, bins=bins, alpha=0.7, color='blue')
    ax.axvline(mean_, c='tab:red', **line_p, label='Mean: %.2f' % mean_)
    ax.axvline(median_, c='tab:green', **line_p, label='Median: %.2f' % median_)
    ax.axvline(mode_, c='tab:orange', **line_p, label='Mode: %.2f' % mode_)
    
    # Add axis information and save figure
    ax.set_xlabel('Criterion length [#characters]')
    ax.set_ylabel('Count')
    

if __name__ == '__main__':
    main()
    