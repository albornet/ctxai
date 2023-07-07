import os
import ast
import pickle
import torch
import torchdata.datapipes.iter as dpi
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from transformers import AutoModel, AutoTokenizer
from cluster_utils import plot_clusters


DATA_DIR = os.path.join('data', 'preprocessed')
LOAD_PATH = os.path.join('data', 'postprocessed')
OUTPUT_PATH = os.path.join('.', 'cluster_plot.png')
LOAD_DATA = False  # True
CHOSEN_PHASES = ['Phase 2', 'Phase 3']
CHOSEN_CONDS = ['Prostate Cancer']
ENCODING = 'utf-8'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(4, os.cpu_count() // 2)
NUM_STEPS = 1718  # max ~= 1718 [* 64 = 110000]
BATCH_SIZE = 64
MODEL_STR_MAP = {
    'pubmed-bert-token': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'bioct-bert-token': 'domenicrosati/ClinicalTrialBioBert-NLI4CT',
    'pubmed-bert-sentence': 'pritamdeka/S-PubMedBert-MS-MARCO',
    'transformer-sentence': 'sentence-transformers/all-mpnet-base-v2',
}
MODEL_TYPE = 'bio-bert-ct'  # see MODEL_STR_MAP.keys()


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
        embeddings, raw_txts, labels = load_data(LOAD_PATH)
    else:
        save_data(LOAD_PATH, embeddings, raw_txts, labels)
        
    # Generate final plot    
    _ = plot_clusters(embeddings, raw_txts, labels, CHOSEN_PHASES, CHOSEN_CONDS)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    

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
    files = dpi.FileLister(data_dir, recursive=True, masks='*.csv')
    sharded = dpi.ShardingFilter(files)  # split file processing by shards
    jsons = dpi.FileOpener(sharded, encoding=ENCODING)
    rows = dpi.CSVParser(jsons)
    data_labels = ClinicalTrialFilter(rows, CHOSEN_PHASES, CHOSEN_CONDS)
    batches = dpi.Batcher(data_labels, batch_size=BATCH_SIZE)
    tokenized_batches = Tokenizer(batches, tokenizer)
    return tokenized_batches


def save_data(path, embeddings, raw_txts, labels):
    """ Saving function
    """
    torch.save(embeddings, os.path.join(path, 'embeddings.pt'))
    with open(os.path.join(path, 'raw_txts.pkl'), 'wb') as f:
        pickle.dump(raw_txts, f)
    with open(os.path.join(path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)


def load_data(path):
    """ Loading function
    """
    embeddings = torch.load(os.path.join(path, 'embeddings.pt'))
    with open(os.path.join(path, 'raw_txts.pkl'), 'rb') as f:
        raw_txts = pickle.load(f)
    with open(os.path.join(path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    return embeddings, raw_txts, labels



class ClinicalTrialFilter(dpi.IterDataPipe):
    def __init__(self,
                 dp: dpi.IterDataPipe,
                 selected_phases: str,
                 selected_conditions: str):
        """ Custom data pipeline to extract input text and label from CT csv file
            and to filter out trials that do not belong to a phase and condition
        """
        self.dp = dp
        all_column_names = next(iter(self.dp))
        cols = ['individual criterion', 'phases', 'conditions', 'category',
                'context', 'subcontext', 'label']
        assert all([c in all_column_names for c in cols])
        self.col_id = {c: all_column_names.index(c) for c in cols}
        self.selected_phases = selected_phases
        self.selected_conditions = selected_conditions
        
    def __iter__(self):
        for i, sample in enumerate(self.dp):
            # Filter out unwanted lines of scv file
            if i == 0: continue
            # if not self._filter_fn(sample): continue
            ct_phases = ast.literal_eval(sample[self.col_id['phases']])
            ct_conditions = ast.literal_eval(sample[self.col_id['conditions']])
            ct_status = sample[self.col_id['label']].lower()
            
            # Yield sample, trying to maximize the number of 'terminated' samples
            labels = {
                'phases': ct_phases,
                'conditions': ct_conditions,
                'status': ct_status
            }
            yield self._build_input_text(sample), labels
            
    def _filter_fn(self, sample):
        """ Filter out CTs that do not belong to a given phase and condition
        """
        ct_phases = ast.literal_eval(sample[self.col_id['phases']])
        ct_conditions = ast.literal_eval(sample[self.col_id['conditions']])
        if all([p not in ct_phases for p in self.selected_phases]):
            return False
        if all([c not in ct_conditions for c in self.selected_conditions]):
            return False
        return True
    
    def _build_input_text(self, sample):
        """ Retrieve criterion and contextual information
        """
        term_sequence = (
            sample[self.col_id['context']],
            sample[self.col_id['subcontext']],
            sample[self.col_id['category']],
            sample[self.col_id['individual criterion']],
        )
        to_join = (s for s in term_sequence if len(s) > 0)
        return ': '.join(to_join).lower()
    
    
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
    