import os
import ast
import csv
import pickle
import torch
import torchdata.datapipes.iter as dpi
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from transformers import AutoModel, AutoTokenizer
from cluster_utils import plot_clusters


INPUT_FORMAT = 'xlsx'  # 'json', 'xlsx'
CSV_FILE_MASK = '*criteria.csv'  # '*criteria.csv', '*example.csv'
LOAD_DATA = False
DATA_DIR = os.path.join('data', 'preprocessed', INPUT_FORMAT)
LOAD_DIR = os.path.join('data', 'postprocessed', INPUT_FORMAT)
RESULT_DIR = os.path.join('results', INPUT_FORMAT)
PLOT_OUTPUT_PATH = os.path.join(RESULT_DIR, 'cluster_plot.png')
STAT_REPORT_OUTPUT_PATH = os.path.join(RESULT_DIR, 'cluster_report_stat.csv')
TEXT_REPORT_OUTPUT_PATH = os.path.join(RESULT_DIR, 'cluster_report_text.csv')
CHOSEN_LABELS = None  # ['completed', 'terminated']  # None to ignore this section filter
CHOSEN_PHASES = None  # ['Phase 2']  # None to ignore this selection filter
CHOSEN_COND_IDS = None  # ['C04']  # None to ignore this selection filter
CHOSEN_ITRV_IDS = None  # ['D02']  # None to ignore this selection filter
ENCODING = 'utf-8'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # 0 for no multiprocessing
NUM_WORKERS = min(NUM_WORKERS, os.cpu_count() // 2)
BATCH_SIZE = 64
MAX_SELECTED_SAMPLES = 8000  # 264000
NUM_STEPS = MAX_SELECTED_SAMPLES // BATCH_SIZE
MODEL_STR_MAP = {
    'pubmed-bert-token': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'bioct-bert-token': 'domenicrosati/ClinicalTrialBioBert-NLI4CT',
    'pubmed-bert-sentence': 'pritamdeka/S-PubMedBert-MS-MARCO',
    'transformer-sentence': 'sentence-transformers/all-mpnet-base-v2',
}
MODEL_TYPE = 'pubmed-bert-sentence'  # anything in MODEL_STR_MAP.keys()


def main():
    # Initialize language model and data pipeline
    if not LOAD_DATA:
        model, tokenizer, pooling_fn = get_model_pipeline(MODEL_TYPE)
        ds = get_dataset(DATA_DIR, tokenizer)
        rs = MultiProcessingReadingService(num_workers=NUM_WORKERS)
        dl = DataLoader2(ds, reading_service=rs)
    
    # Populate tensor with eligibility criteria embeddings
    if not LOAD_DATA:
        raw_txts, labels = [], []
        embeddings = torch.empty((0, model.config.hidden_size))
        for i, (encoded, raw_txt, label) in tqdm(enumerate(dl), total=NUM_STEPS):
            
            # Compute embeddings for this batch
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            sent_embeddings = pooling_fn(encoded, outputs)
            
            # Populate tensor with the model outputs, input texts and labels
            embeddings = torch.cat((embeddings, sent_embeddings.cpu()), dim=0)
            raw_txts.extend(raw_txt)
            labels.extend(label)
            if i >= NUM_STEPS - 1: break
    
    # Save / load data
    if LOAD_DATA:
        embeddings, raw_txts, labels = load_data(LOAD_DIR, MODEL_TYPE)
    else:
        save_data(LOAD_DIR, MODEL_TYPE, embeddings, raw_txts, labels)
        
    # Generate final plot and final csv report
    _, text_report, stat_report = plot_clusters(embeddings, raw_txts, labels)
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    with open(STAT_REPORT_OUTPUT_PATH, 'w', newline='') as file:
        csv.writer(file).writerows(stat_report)
    with open(TEXT_REPORT_OUTPUT_PATH, 'w', newline='') as file:
        csv.writer(file).writerows(text_report)
    

def get_model_pipeline(model_type):
    """ Select a model and the corresponding tokenizer and embed function
    """
    # Model generates token-level embeddings, and output [cls] (+ linear + tanh)
    if model_type in ['pubmed-bert-token', 'bioct-bert-token']:
        def pooling_fn(encoded_input, model_output):
            return model_output['pooler_output']

    # Model generates sentence-level embeddings directly
    elif model_type in ['pubmed-bert-sentence', 'transformer-sentence']:
        def pooling_fn(encoded_input, model_output):
            token_embeddings = model_output[0]
            attn_mask = encoded_input['attention_mask'].unsqueeze(-1)
            input_expanded = attn_mask.expand(token_embeddings.size()).float()
            token_sum = torch.sum(token_embeddings * input_expanded, dim=1)
            return token_sum / torch.clamp(input_expanded.sum(1), min=1e-9)
            
    # Return model (sent to correct device) and tokenizer
    model_str = MODEL_STR_MAP[model_type]
    model = AutoModel.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return model.to(DEVICE), tokenizer, pooling_fn


def get_dataset(data_dir, tokenizer):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    files = dpi.FileLister(data_dir, recursive=True, masks=CSV_FILE_MASK)
    sharded = dpi.ShardingFilter(files)  # split file processing by shards
    jsons = dpi.FileOpener(sharded, encoding=ENCODING)
    rows = dpi.CSVParser(jsons)
    data_labels = ClinicalTrialFilter(rows, CHOSEN_LABELS, CHOSEN_PHASES,
                                      CHOSEN_COND_IDS, CHOSEN_ITRV_IDS)
    batches = dpi.Batcher(data_labels, batch_size=BATCH_SIZE)
    tokenized_batches = Tokenizer(batches, tokenizer)
    return tokenized_batches


def save_data(dir_, model_type, embeddings, raw_txts, labels):
    """ Simple saving function for model predictions
    """
    os.makedirs(dir_, exist_ok=True)
    torch.save(embeddings, os.path.join(dir_, 'embeddings_%s.pt' % model_type))
    with open(os.path.join(dir_, 'raw_txts.pkl'), 'wb') as f:
        pickle.dump(raw_txts, f)
    with open(os.path.join(dir_, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)


def load_data(dir_, model_type):
    """ Simple loading function for model predictions
    """
    embeddings = torch.load(os.path.join(dir_, 'embeddings_%s.pt' % model_type))
    with open(os.path.join(dir_, 'raw_txts.pkl'), 'rb') as f:
        raw_txts = pickle.load(f)
    with open(os.path.join(dir_, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    return embeddings, raw_txts, labels


class ClinicalTrialFilter(dpi.IterDataPipe):
    def __init__(self,
                 dp: dpi.IterDataPipe,
                 chosen_labels: list[str],
                 chosen_phases: list[str],
                 chosen_cond_ids: list[str],
                 chosen_itrv_ids: list[str],
                 ) -> None:
        """ Custom data pipeline to extract input text and label from CT csv file
            and to filter out trials that do not belong to a phase and condition
        """
        self.dp = dp
        all_column_names = next(iter(self.dp))
        cols = ['individual criterion', 'phases', 'ct path', 'condition_ids',
                'intervention_ids', 'category', 'context', 'subcontext', 'label']
        assert all([c in all_column_names for c in cols])
        self.col_id = {c: all_column_names.index(c) for c in cols}
        self.chosen_labels = chosen_labels
        self.chosen_phases = chosen_phases
        self.chosen_cond_ids = chosen_cond_ids
        self.chosen_itrv_ids = chosen_itrv_ids
        
    def __iter__(self):
        for i, sample in enumerate(self.dp):
            # Filter out unwanted lines of the csv file
            if i == 0: continue
            ct_path = sample[self.col_id['ct path']],
            ct_status = sample[self.col_id['label']].lower()
            if not self._filter_fn(sample, ct_status): continue
            
            # Yield sample and labels if all is good
            labels = {'ct_path': ct_path, 'ct_status': ct_status}
            yield self._build_input_text(sample), labels
            
    def _filter_fn(self, sample, label):
        """ Filter out CTs that do not belong to a given phase and condition
        """
        # Load relevant data (condition and intervention ids)
        ct_phases = ast.literal_eval(sample[self.col_id['phases']])
        ct_cond_ids = ast.literal_eval(sample[self.col_id['condition_ids']])
        ct_itrv_ids = ast.literal_eval(sample[self.col_id['intervention_ids']])
        ct_cond_ids = [c for cc in ct_cond_ids for c in cc]  # flatten
        ct_itrv_ids = [i for ii in ct_itrv_ids for i in ii]  # flatten
        
        # Check criterion's phase
        if self.chosen_phases is not None\
        and all([p not in ct_phases for p in self.chosen_phases]):
            return False
        
        # Check criterion's conditions
        if self.chosen_cond_ids is not None\
        and all([not any([c.startswith(i) for i in self.chosen_cond_ids])
                 for c in ct_cond_ids]):
            return False
        
        # Check criterion's interventions
        if self.chosen_itrv_ids is not None\
        and all([not any([c.startswith(i) for i in self.chosen_itrv_ids])
                 for c in ct_itrv_ids]):
            return False
        
        # Check criterion's status
        if self.chosen_labels is not None and label not in self.chosen_labels:
            return False
        
        # Accept to yield criterion if it passes all filters
        return True
    
    def _build_input_text(self, sample):
        """ Retrieve criterion and contextual information
        """
        term_sequence = (
            sample[self.col_id['category']] + 'clusion criterion',
            # sample[self.col_id['context']],
            sample[self.col_id['subcontext']],
            sample[self.col_id['individual criterion']],
        )
        to_join = (s for s in term_sequence if len(s) > 0)
        return ' - '.join(to_join).lower()
    
    
class Tokenizer(dpi.IterDataPipe):
    def __init__(self, dp: dpi.IterDataPipe, tokenizer):
        """ Custom data pipeline to tokenize an batch of input strings, keeping
            the corresponding labels and returning input and label batches
        """
        self.dp = dp
        self.tokenizer = tokenizer
        
    def __iter__(self):
        for batch in self.dp:
            input_batch, label_batch = zip(*batch)
            yield self.tokenize_fn(input_batch), input_batch, label_batch
    
    def tokenize_fn(self, batch):
        return self.tokenizer(batch,
                              padding=True,
                              truncation=True,
                              max_length=512,
                              return_tensors='pt')
            
            
if __name__ == '__main__':
    main()
    