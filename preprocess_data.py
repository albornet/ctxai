import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torchdata.datapipes.iter as dpi
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from parsing_utils import ClinicalTrialFilter, CriteriaParser, CriteriaCSVWriter
from statistics import mean, stdev, median, mode
from tqdm import tqdm
from typing import Union


DATA_DIR = os.path.join('.', 'data', 'ALLAPIJSON')
RESULT_DIR = os.path.join('.', 'data', 'preprocessed')
PLOT_PATH = os.path.join(RESULT_DIR, 'criteria_length_histogram.png')
CSV_PATH = os.path.join(RESULT_DIR, 'parsed_criteria.csv')
CSV_HEADERS = [
    'criteria paragraph', 'complexity', 'ct path', 'label', 'phases',
    'conditions', 'condition_ids', 'intervention_ids', 'category', 'context',
    'subcontext', 'individual criterion',
]
ENCODING = 'utf-8'
NUM_STEPS = 110000  # actually 109685 in total
NUM_WORKERS = 12
NUM_WORKERS = min(NUM_WORKERS, max(os.cpu_count() - 4, os.cpu_count() // 4))
LOAD_CSV_RESULTS = False
EXAMPLE = True
DEBUG = False
if DEBUG:
    EXAMPLE = True
    NUM_WORKERS = 0
    LOAD_CSV_RESULTS = False
if EXAMPLE:
    NUM_STEPS = 1000
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
        ds = get_dataset(shuffle=True)  # parse CT into individual criteria lists
        rs = MultiProcessingReadingService(num_workers=NUM_WORKERS)
        dl = DataLoader2(ds, reading_service=rs)
        
        # Write parsed criteria to the output file
        num_steps = min(NUM_STEPS, total_number_of_files_to_process(DATA_DIR))
        for i, data in tqdm(enumerate(dl),
                            total=num_steps,
                            desc='Parsing criteria'):
            with open(CSV_PATH, 'a', newline='', encoding=ENCODING) as f:
                writer = csv.writer(f)
                writer.writerows(data)
            if i >= num_steps - 1: break
        dl.shutdown()
    
    # Evaluate csv result file, and compute some statistics
    compute_result_stats(CSV_PATH)
    
    
def total_number_of_files_to_process(data_dir=DATA_DIR):
    """ Compute the total number of files to be processed by the data pipeline
    """
    count = 0
    for _, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith('.json'): count += 1
    return count
    
    
def get_dataset(data_dir=DATA_DIR, shuffle=False):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    files = dpi.FileLister(data_dir, recursive=True, masks='*.json')
    sharded = dpi.ShardingFilter(files)  # split file processing by shards
    if shuffle: sharded = dpi.Shuffler(sharded)
    jsons = dpi.FileOpener(sharded, encoding=ENCODING)
    dicts = dpi.JsonParser(jsons)
    raw_samples = ClinicalTrialFilter(dicts)
    parsed_samples = CriteriaParser(raw_samples)
    written = CriteriaCSVWriter(parsed_samples)
    return written

    
def compute_result_stats(result_path: str,
                         plot_path=PLOT_PATH,
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
    