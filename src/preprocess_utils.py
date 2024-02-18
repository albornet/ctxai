# Config
import os
try:
    import config
except:
    from . import config
import logging
cfg = config.get_config()
logger = logging.getLogger("cluster")

# Utils
import re
import ast
import json
import random
import numpy as np
import pandas as pd
import torch
import gc
import cupy as cp

# Data pipelines
import nltk
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
# from clinitokenizer.tokenize import clini_tokenize
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


EXCLUSION_KEYS = [s + '\:?\.?' for s in [
    'exclusion criteria', 'exclusion criterion', 'exclusions?', 'excluded',
    'ineligible', 'not eligible', 'not allowed', 'must not have', 'must not be',
    'patients? have no ', 'patients? have not ', 'patients? had no ',
    'patients? had not ', 'patients? must not ', 'no patients?',
]]
INCLUSION_KEYS = [s + '\:?\.?' for s in [
    'inclusion criteria', 'inclusion criterion', 'inclusions?', 'included',
    'eligible', 'allowed', 'must have', 'must be', 'patients? have ',
    'patients? had ', 'had to have', 'required', 'populations? consisted',
    'not excluded', 'not be excluded',
]]
CONTEXT_KEYS = [
    'prior medications?', 'prior treatments?', 'concurrent medications?',
    'weight restrictions?', 'social habits?', 'medications?',  # 'diseases?'
    'concurrent treatments?', 'co-existing conditions?', 'risk behaviors?',
    'prior concurrent therapy', 'prior concurrent therapies', 'recommended',
    'medical complications?', 'obstetrical complications?', 'group a', 'group b',
    'part a', 'part b', 'phase a', 'phase b', 'phase i', 'phase ii', 'phase iii',
    'phase iv', 'discouraged', 'avoid', 'patients?', 'patient characteristics',
    'disease characteristics', 'elevated psa criteria', 'psa criteria',
    'initial doses?', 'additional doses?',
]
SUBCONTEXT_KEYS = [
    'infants?', 'allowed for infants?', 'women', 'allowed for women',
    'allowed for pregnant women', 'life expectancy', 'hematopoietic',
    'hematologic', 'hepatic', 'renal', 'cardiovascular', 'cardiac', 'pulmonary',
    'systemic', 'biologic therapy', 'chemotherapy', 'endocrine therapy',
    'radiotherapy', 'surgery', 'other', 'performance status', 'age', 'sex',
    'definitive local therapy', 'previous hormonal therapy or other treatments?',
    'other treatments?', 'previous hormonal therapy', 'in either eyes?',
    'in study eyes?', 'one of the following', 'body mass index', 'bmi',
    'eligible subtypes?', 'hormone receptor status', 'menopausal status',
    'at least 1 of the following factors', 'serology', 'chemistry',
]
SIMILARITY_FN = lambda s, k: SequenceMatcher(None, s, k).ratio()
MAX_SIMILARITY_FN = lambda s, keys: max([SIMILARITY_FN(s.lower(), k) for k in keys])


@functional_datapipe('filter_clinical_trials')
class ClinicalTrialFilter(IterDataPipe):
    """ Read clinical trial json files, parse them and filter the ones that can
        be used for eligibility criteria representation learning
    """
    def __init__(self, dp):
        super().__init__()
        self.dp = dp
        mesh_cw_path = os.path.join(cfg["BASE_DATA_DIR"], cfg["MESH_CROSSWALK_NAME"])
        with open(mesh_cw_path, 'r') as f:
            self.mesh_cw = json.load(f)
            
    def __iter__(self):
        for ct_path, ct_dict in self.dp:
            # Load protocol and make sure it corresponds to the file name
            protocol = ct_dict['FullStudy']['Study']['ProtocolSection']
            derived = ct_dict['FullStudy']['Study']['DerivedSection']
            nct_id = protocol['IdentificationModule']['NCTId']
            assert nct_id == os.path.split(ct_path)[-1].strip('.json')
            
            # Check protocol belongs to the data and load criteria
            good_to_go, label, phases, conditions, cond_ids, itrv_ids =\
                self.check_protocol(protocol, derived)
            if good_to_go:
                metadata = {
                    'ct_path': ct_path,
                    'label': label,
                    'phases': phases,
                    'conditions': conditions,
                    'condition_ids': cond_ids,
                    'intervention_ids': itrv_ids,
                }
                criteria_str = protocol['EligibilityModule']['EligibilityCriteria']
                yield metadata, criteria_str
                
    def check_protocol(self, protocol, derived):
        """ Parse clinical trial protocol and make sure it can be used as a data
            sample for eligibility criteria representation learning 
        """
        # Check the status of the CT is either completed or terminated
        status = protocol['StatusModule']['OverallStatus']
        if status not in ['Completed', 'Terminated']:
            return False, None, None, None, None, None
        
        # Check that the study is interventional
        study_type = protocol['DesignModule']['StudyType']
        if study_type != 'Interventional':
            return False, None, None, None, None, None
        
        # Check the study is about a drug test
        interventions = protocol[
            'ArmsInterventionsModule']['InterventionList']['Intervention']
        intervention_types = [i['InterventionType'] for i in interventions]
        if 'Drug' not in intervention_types:
            return False, None, None, None, None, None
        
        # Check the study has defined phases, then record phases
        if 'PhaseList' not in protocol['DesignModule']:
            return False, None, None, None, None, None
        phases = protocol['DesignModule']['PhaseList']['Phase']
        
        # Check that the protocol has an eligibility criterion section
        if 'EligibilityCriteria' not in protocol['EligibilityModule']:
            return False, None, None, None, None, None
        
        # Check that the protocol has a condition list
        if 'ConditionList' not in protocol['ConditionsModule']:
            return False, None, None, None, None, None
        conditions = protocol['ConditionsModule']['ConditionList']['Condition']
        
        # Try to load condition mesh ids
        try:
            conds = derived['ConditionBrowseModule']['ConditionMeshList']
            cond_ids = [c['ConditionMeshId'] for c in conds['ConditionMesh']]
        except KeyError:
            cond_ids = []
        cond_treenums = self.convert_unique_ids_to_tree_nums(cond_ids)
                
        # Try to load intervention mesh ids
        try:
            itrvs = derived['InterventionBrowseModule']['InterventionMeshList']
            itrv_ids = [i['InterventionMeshId'] for i in itrvs['InterventionMesh']]
        except KeyError:
            itrv_ids = []
        itrv_tree_nums = self.convert_unique_ids_to_tree_nums(itrv_ids)
        
        # Return that the protocol can be processed, status, and phase list
        return True, status, phases, conditions, cond_treenums, itrv_tree_nums
    
    def convert_unique_ids_to_tree_nums(self, unique_ids):
        """ Try to convert a maximum of unique id found in a clinical trial to
            its tree num counterpart, solving the trailing zeros problem
        """
        tree_nums = []
        for i in unique_ids:
            try:
                tree_nums.append(self.mesh_cw[i.replace('000', '', 1)])
            except KeyError:
                try:
                    tree_nums.append(self.mesh_cw[i])
                except KeyError:
                    pass
        return tree_nums
    
    
@functional_datapipe('parse_criteria')
class CriteriaParser(IterDataPipe):
    """ Parse criteria raw paragraphs into a set of individual criteria, as well
        as other features and labels, such as medical context
    """
    def __init__(self, dp):
        super().__init__()
        logger.info(" - Downloading package punkt to tokenize sentences")
        nltk.download("punkt", quiet=True)
        self.dp = dp
    
    def __iter__(self):
        for metadata, criteria_str in self.dp:
            parsed_criteria, complexity = self.parse_criteria(criteria_str)
            metadata['complexity'] = complexity
            metadata['criteria_str'] = criteria_str
            yield metadata, parsed_criteria
            
    def parse_criteria(self,
                       criteria_str: str,
                       is_section_title_thresh: float=0.6,
                       is_bug_thresh: int=5,
                       ) -> list[dict[str, str]]:
        """ Parse a criteria paragraph into a set of criterion sentences,
            categorized as inclusion, exclusion, or undetermined criterion
        """
        # Split criteria paragraph into a set of sentences
        paragraphs = [s.strip() for s in criteria_str.split('\n') if s.strip()]
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(self.split_by_period(paragraph))
            
        # Initialize running variables and go through every sentence
        parsed = []
        prev_category = '?'
        similarity_threshold = 0.0
        for sentence in sentences:
            
            # Match sentence to exclusion and inclusion key expressions
            found_in = any(re.search(k, sentence, re.IGNORECASE) for k in INCLUSION_KEYS)
            found_ex = any(re.search(k, sentence, re.IGNORECASE) for k in EXCLUSION_KEYS)
            if re.search('not (be )?excluded', sentence, re.IGNORECASE):
                found_ex = False  # special case (could do better?)
            
            # Compute max similarity with any key, and if a prev update is needed
            key_similarity = MAX_SIMILARITY_FN(sentence, INCLUSION_KEYS + EXCLUSION_KEYS)
            should_update_prev = key_similarity > similarity_threshold
            
            # Based on the result, determine sentence and prev categories
            category, prev_category = self.categorise_sentence(
                found_ex, found_in, prev_category, should_update_prev)
            
            # Add criterion to the list only if it is not a section title
            sentence_is_section_title = key_similarity > is_section_title_thresh
            if sentence_is_section_title:
                similarity_threshold = is_section_title_thresh
            else:
                parsed.append({'category': category, 'text': sentence})
                
        # Try to further split parsed criteria, using empirical methods
        parsed = self.contextualize_criteria(parsed)
        parsed = self.post_process_criteria(parsed)
        
        # Return final list of criteria, as well as how easy it was to split
        parsed = [c for c in parsed if len(c['text']) >= is_bug_thresh]
        complexity = 'easy' if similarity_threshold > 0 else 'hard'
        return parsed, complexity
    
    @staticmethod
    def split_by_period(text: str) -> list[str]:
        """ Sentence tokenizer does bad with "e.g." and "i.e.", hence a special
            function that helps it a bit (any other edge-case to add?)
        """
        text = text.replace('e.g.', 'e_g_')\
                   .replace('i.e.', 'i_e_')\
                   .replace('etc.', 'etc_')
        splits = [s.strip() for s in sent_tokenize(text) if s.strip()]
        return [s.replace('e_g_', 'e.g.')
                 .replace('i_e_', 'i.e.')
                 .replace('etc_', 'etc.') for s in splits]
    
    @staticmethod
    def categorise_sentence(found_ex: bool,
                            found_in: bool,
                            prev_category: str,
                            should_update_prev: bool,
                            ) -> tuple[str, str]:
        """ Categorize a sentence based on the following parameters:
            - found_ex: whether an exclusion phrase was matched to the sentence
            - found_in: whether an inclusion phrase was matched to the sentence
            - prev_category: one previous category that may help determine
                the current sentence, in case no expression was matched to it
            - should_update_prev: whether current sentence should be used to
                update previous category
        """
        # If a key expression was matched, try to update prev category
        if found_ex:  # has to be done before found_in!
            category = 'ex'
            if should_update_prev: prev_category = 'ex'
        elif found_in:
            category = 'in'
            if should_update_prev: prev_category = 'in'
        
        # If no key expression was matched, use prev category
        else:
            category = prev_category
            
        # Return category and updated (or not) prev category
        return category, prev_category
    
    @staticmethod
    def contextualize_criteria(parsed_criteria: list[dict[str, str]],
                               is_context_thresh: float=0.9,
                               is_subcontext_thresh: float=0.8,
                               ) -> list[dict[str, str]]:
        """ Try to add context to all criteria identified by using keys that tend
            to appear as subsections of inclusion/exclusion criteria.
            The keys are also used to split criteria when they are not written
            with newlines (mere string, but including context keys)
        """
        # Initialize variables and go through all parsed criteria
        contextualized = []
        context, subcontext, prev_category = '', '', ''
        for criterion in parsed_criteria:
            
            # Split criterion by any context or subcontext keys, keeping matches
            sentence, category = criterion['text'], criterion['category']
            pattern = "|".join(['(%s(?=\:))' % k for k in CONTEXT_KEYS + SUBCONTEXT_KEYS])
            splits = re.split(pattern, sentence, flags=re.IGNORECASE)
            for split in [s for s in splits if s is not None]:
                
                # If any split corresponds to a context key, define it as context
                if MAX_SIMILARITY_FN(split, CONTEXT_KEYS) > is_context_thresh\
                    and not 'see ' in split.lower():
                    context = split.strip(':')
                    subcontext = ''  # reset subcontext if new context
                    continue
                
                # Same, but for subcontext keys
                if MAX_SIMILARITY_FN(split, SUBCONTEXT_KEYS) > is_subcontext_thresh\
                    and not 'see ' in split.lower():
                    subcontext = split.strip(':')
                    continue
                
                # Append non-matching criterion, with previous (sub)context match
                contextualized.append({
                    'category': criterion['category'],
                    'context': context,
                    'subcontext': subcontext,
                    'text': split.strip('\n\t :'),
                })
            
            # # Small check in case previous category was different (ok?)
            # if category != prev_category:
            #     context, subcontext = '', ''
            #     prev_category = category
            
        # Return newly parsed set of criteria, with added context
        return contextualized
        
    @staticmethod
    def post_process_criteria(parsed_criteria: list[dict[str, str]],
                              placeholder='*'
                              ) -> list[dict[str, str]]:
        """ Split each criterion by semicolon, avoiding false positives, such as
            within parentheses or quotation marks, also remove '<br>n' bugs
        """
        post_parsed = []
        for criterion in parsed_criteria:
            # Replace false positive semicolon separators by '*' characters
            regex = r'\([^)]*\)|\[[^\]]*\]|"[^"]*"|\*[^*]*\*|\'[^\']*\''
            replace_fn = lambda match: match.group(0).replace(';', placeholder)
            hidden_criterion = re.sub(regex, replace_fn, criterion['text'])
            hidden_criterion = re.sub(r'<br>\d+\)?\s*', '', hidden_criterion)
            
            # Split by semicolon and put back semicolons that were protected
            splits = hidden_criterion.split(';')
            splits = [split.replace(placeholder, ';') for split in splits]        
            post_parsed.extend([dict(criterion, text=s.strip()) for s in splits])
        
        # Return post-processed criteria
        return post_parsed
        
        
@functional_datapipe('write_csv')
class CriteriaCSVWriter(IterDataPipe):
    """ Take the output of CriteriaParser (list of dictionaries) and transform
        it into a list of lists of strings, ready to be written to a csv file
    """
    def __init__(self, dp: IterDataPipe) -> None:
        super().__init__()
        self.dp = dp
        
    def __iter__(self):
        for metadata, parsed_criteria in self.dp:
            yield self.generate_csv_rows(metadata, parsed_criteria)
            
    @staticmethod
    def generate_csv_rows(metadata: dict[str, str],
                          parsed_criteria: list[dict[str, str]]
                          ) -> list[list[str]]:
        """ Generate a set of rows to be written to a csv file
        """
        return [[
            metadata['criteria_str'] if i == 0 else '',
            metadata['complexity'] if i == 0 else '',
            metadata['ct_path'],
            metadata['label'],
            metadata['phases'],
            metadata['conditions'],
            metadata['condition_ids'],
            metadata['intervention_ids'],
            c['category'],
            c['context'],
            c['subcontext'],
            c['text'].replace('≥', '> or equal to')
                     .replace('≤', '< or equal to')
                     .strip('- '),
        ] for i, c in enumerate(parsed_criteria)]


@functional_datapipe('read_xlsx_lines')
class CustomXLSXLineReader(IterDataPipe):
    def __init__(self, dp):
        """ Lalalala
        """
        self.dp = dp
        self.metadata_mapping = {
            'trialid': 'ct_path',
            'recruitmentStatusNorm': 'label',
            'phaseNorm': 'phases',
            'conditions_': 'conditions',
            'conditions': 'condition_ids',
            'interventions': 'intervention_ids',
        }
        mesh_cw_path = os.path.join(cfg["BASE_DATA_DIR"], cfg["MESH_CROSSWALK_NAME"])
        with open(mesh_cw_path, 'r') as f:
            self.mesh_cw = json.load(f)
        self.intervention_remove = [
            'Drug: ', 'Biological: ', 'Radiation: ',
            'Procedure: ', 'Other: ', 'Device: ',
        ]
    
    def __iter__(self):
        for file_name in self.dp:
            sheet_df = pd.ExcelFile(file_name).parse('metadata')
            crit_str_list = self.extract_criteria_strs(sheet_df)
            metatdata_dict_list = self.extract_metadata_dicts(sheet_df)
            for crit_str, metadata in zip(crit_str_list, metatdata_dict_list):
                yield metadata, crit_str
    
    @staticmethod
    def extract_criteria_strs(sheet_df: pd.DataFrame) -> list[str]:
        """ Extract criteria text from the dataframe
        """
        in_crit_strs = sheet_df['inclusionCriteriaNorm'].fillna('')
        ex_crit_strs = sheet_df['exclusionCriteriaNorm'].fillna('')
        crit_strs = in_crit_strs + '\n\n' + ex_crit_strs
        return crit_strs
    
    def extract_metadata_dicts(self, sheet_df: pd.DataFrame) -> list[dict]:
        """ Extract metadata information for each criteria text
        """
        sheet_df['conditions_'] = sheet_df['conditions']
        sheet_df = sheet_df[list(self.metadata_mapping.keys())]
        sheet_df = sheet_df.rename(columns=self.metadata_mapping)
        split_fn = lambda s: s.split('; ')
        map_fn = lambda l: self.convert_unique_ids_to_tree_nums(l)
        sheet_df['conditions'] = sheet_df['conditions'].apply(split_fn)
        sheet_df['condition_ids'] = sheet_df['condition_ids'].apply(split_fn)
        sheet_df['condition_ids'] = sheet_df['condition_ids'].apply(map_fn)
        sheet_df['intervention_ids'] = sheet_df['intervention_ids'].apply(split_fn)
        sheet_df['intervention_ids'] = sheet_df['intervention_ids'].apply(map_fn)
        list_of_metadata = sheet_df.to_dict('records')
        return list_of_metadata
    
    def convert_unique_ids_to_tree_nums(self, unique_names):
        """ Try to convert a maximum of unique names found in a clinical trial to
            its tree num counterpart, solving the trailing zeros problem
        """
        for r in self.intervention_remove:
            unique_names = [n.replace(r, '') for n in unique_names]
        tree_nums = []
        for n in unique_names:
            try:
                tree_nums.append(self.mesh_cw[n.replace('000', '', 1)])
            except KeyError:
                try:
                    tree_nums.append(self.mesh_cw[n])
                except KeyError:
                    pass
        return tree_nums
    

@functional_datapipe('read_dict_lines')
class CustomDictLineReader(IterDataPipe):
    def __init__(self, dp):
        """ Lalalala
        """
        self.dp = dp
        self.metadata_mappings = {
            'trialid': 'ct_path',
            'cluster': 'label',
            'sentence_preprocessed': 'condition_ids',
        }
    
    def __iter__(self):
        for file_name in self.dp:
            sheet_df = pd.ExcelFile(file_name).parse('Sheet1')
            crit_str_list = self.extract_criteria_strs(sheet_df)
            metatdata_dict_list = self.extract_metadata_dicts(sheet_df)
            for crit_str, metadata in zip(crit_str_list, metatdata_dict_list):
                yield metadata, crit_str
    
    def extract_criteria_strs(self, sheet_df: pd.DataFrame) -> list[str]:
        """ Extract eligibility criteria from the dataframe
        """
        sheet_df = sheet_df['sentence']
        sheet_df = sheet_df.apply(self.strip_fn)
        sheet_df = sheet_df.apply(self.criterion_format_fn)
        return sheet_df  # .to_list()
    
    @staticmethod
    def strip_fn(s: str):
        """ Remove trailing spaces for a criteria
        """
        return s.strip()
    
    @staticmethod
    def criterion_format_fn(criteria_str: pd.DataFrame) -> pd.DataFrame:
        criteria_dict = {
            'category': '',
            'context': '',
            'subcontext': '',
            'text': criteria_str
        }
        return [criteria_dict]
    
    def extract_metadata_dicts(self, sheet_df: pd.DataFrame) -> list[dict]:
        """ Extract metadata information for each criterion
        """
        sheet_df = sheet_df.filter(self.metadata_mappings.keys())
        sheet_df = sheet_df.rename(self.metadata_mappings, axis=1)
        sheet_df['criteria_str'] = sheet_df.apply(lambda _: '', axis=1)
        sheet_df['complexity'] = sheet_df.apply(lambda _: '', axis=1)
        sheet_df['phases'] = sheet_df.apply(lambda _: [''], axis=1)
        sheet_df['conditions'] = sheet_df.apply(lambda _: [''], axis=1)
        sheet_df['intervention_ids'] = sheet_df.apply(lambda _: [''], axis=1)
        sheet_df['condition_ids'] = sheet_df['condition_ids'].apply(
            lambda s: [s.strip()],
        )
        list_of_metadata = sheet_df.to_dict('records')
        return list_of_metadata


@functional_datapipe("filter_eligibility_criteria")
class EligibilityCriteriaFilter(IterDataPipe):
    def __init__(
        self,
        dp: IterDataPipe,
        chosen_phases: list[str]=cfg["CHOSEN_PHASES"],
        chosen_statuses: list[str]=cfg["CHOSEN_STATUSES"],
        chosen_criteria: list[str]=cfg["CHOSEN_CRITERIA"],
        chosen_cond_ids: list[str]=cfg["CHOSEN_COND_IDS"],
        chosen_itrv_ids: list[str]=cfg["CHOSEN_ITRV_IDS"],
        chosen_cond_lvl: int=cfg["CHOSEN_COND_LVL"],
        chosen_itrv_lvl: int=cfg["CHOSEN_ITRV_LVL"],        
    ) -> None:
        """ Data pipeline to extract text and labels from a csv file containing
            eligibility criteria and to filter out samples whose clinical trial
            does not include a set of statuses/phases/conditions/interventions
        """
        # Initialize filters
        self.chosen_phases = chosen_phases
        self.chosen_statuses = chosen_statuses
        self.chosen_criteria = chosen_criteria
        self.chosen_cond_ids = chosen_cond_ids
        self.chosen_itrv_ids = chosen_itrv_ids
        self.chosen_cond_lvl = chosen_cond_lvl
        self.chosen_itrv_lvl = chosen_itrv_lvl
        
        # Load crosswalk between mesh terms and conditions / interventions
        mesh_cw_inverted_path = \
            os.path.join(cfg["BASE_DATA_DIR"], cfg["MESH_CROSSWALK_INVERTED_NAME"])
        with open(mesh_cw_inverted_path, "r") as f:
            self.mesh_cw_inverted = json.load(f)
        
        # Initialize data pipeline
        self.dp = dp
        all_column_names = next(iter(self.dp))
        cols = [
            "individual criterion", "phases", "ct path", "condition_ids",
            "intervention_ids", "category", "context", "subcontext", "label",
        ]
        assert all([c in all_column_names for c in cols])
        self.col_id = {c: all_column_names.index(c) for c in cols}
        self.yielded_input_texts = []
        
    def __iter__(self):
        for i, sample in enumerate(self.dp):
            
            # Filter out unwanted lines of the csv file
            if i == 0: continue
            ct_metadata, ct_not_filtered = self._filter_fn(sample)
            if not ct_not_filtered: continue
            
            # Yield sample and metadata ("labels") if all is good
            input_text = self._build_input_text(sample)
            if input_text not in self.yielded_input_texts:
                self.yielded_input_texts.append(input_text)
                yield input_text, ct_metadata
            
    def _filter_fn(self, sample: dict[str, str]):
        """ Filter out eligibility criteria whose clinical trial does not include
            a given set of statuses/phases/conditions/interventions
        """
        # Initialize metadata
        ct_path = sample[self.col_id["ct path"]],
        ct_status = sample[self.col_id["label"]].lower()
        metadata = {"path": ct_path, "status": ct_status}
        
        # Load relevant data
        ct_phases = ast.literal_eval(sample[self.col_id["phases"]])
        ct_cond_ids = ast.literal_eval(sample[self.col_id["condition_ids"]])
        ct_itrv_ids = ast.literal_eval(sample[self.col_id["intervention_ids"]])
        ct_cond_ids = [c for cc in ct_cond_ids for c in cc]  # flatten
        ct_itrv_ids = [i for ii in ct_itrv_ids for i in ii]  # flatten
        ct_category = sample[self.col_id["category"]]
        
        # Check criterion phases
        if len(self.chosen_phases) > 0:
            if all([p not in self.chosen_phases for p in ct_phases]):
                return metadata, False
        
        # Check criterion conditions
        cond_lbls = self._get_cond_itrv_labels(
            ct_cond_ids, self.chosen_cond_ids, self.chosen_cond_lvl)
        if len(cond_lbls) == 0:
            return metadata, False
        
        # Check criterion interventions
        itrv_lbls = self._get_cond_itrv_labels(
            ct_itrv_ids, self.chosen_itrv_ids, self.chosen_itrv_lvl)
        if len(itrv_lbls) == 0:
            return metadata, False
        
        # Check criterion status
        if len(self.chosen_statuses) > 0:
            if ct_status not in self.chosen_statuses:
                return metadata, False
        
        # Check criterion type
        if len(self.chosen_criteria) > 0:
            if ct_category not in self.chosen_criteria:
                return metadata, False
        
        # Update metadata
        metadata["phase"] = ct_phases
        metadata["condition_ids"] = ct_cond_ids
        metadata["condition"] = cond_lbls
        metadata["intervention_ids"] = ct_itrv_ids
        metadata["intervention"] = itrv_lbls
        metadata["label"] = self._get_unique_label(ct_phases, cond_lbls, itrv_lbls)

        # Accept to yield criterion if it passes all filters
        return metadata, True
    
    @staticmethod
    def _get_unique_label(
        phases: list[str],
        cond_lbls: list[str],
        itrv_lbls: list[str]
    ) -> str:
        """ Build a single label for any combination of phase, condition, and
            intervention
        """
        phase_lbl = " - ".join(sorted(phases))
        cond_lbl = " - ".join(sorted(cond_lbls))
        itrv_lbl = " - ".join(sorted(itrv_lbls))
        return " --- ".join([phase_lbl, cond_lbl, itrv_lbl])
    
    def _get_cond_itrv_labels(
        self,
        ct_ids: list[str],
        chosen_ids: list[str],
        level: int,
    ) -> list[str]:
        """ Construct a list of unique mesh tree labels for a list of condition
            or intervention mesh codes, aiming a specific level in the hierachy
        """
        # Case where condition or intervention is not important
        if level is None: return ["N/A"]
        
        # Filter condition or intervention ids
        if len(chosen_ids) > 0:
            ct_ids = [c for c in ct_ids if any([c.startswith(i) for i in chosen_ids])]
        
        # Select only the ones that have enough depth
        n_chars = level * 4 - 1  # format: at least "abc.def.ghi.jkl"
        cut_ct_ids = [c[:n_chars] for c in ct_ids if len(c.split(".")) >= level]
        
        # Map ids to non-code labels
        labels = [self.mesh_cw_inverted[c] for c in cut_ct_ids]
        is_a_code = lambda lbl: (sum(c.isdigit() for c in lbl) + 1 == len(lbl))
        labels = [l for l in labels if not is_a_code(l)]
        
        # Return unique values
        return list(set(labels))
    
    def _build_input_text(self, sample):
        """ Retrieve criterion and contextual information
        """
        term_sequence = (
            sample[self.col_id["category"]] + "clusion criterion",
            # sample[self.col_id["context"]],
            # sample[self.col_id["subcontext"]],
            sample[self.col_id["individual criterion"]],
        )
        to_join = (s for s in term_sequence if len(s) > 0)
        return " - ".join(to_join).lower()


def set_seeds(seed_value=1234):
    """ Set seed for reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If using PyTorch and you want determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_memory_fn():
    """ Try to remove unused variables in GPU and CPU, after each model run
    """
    gc.collect()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def clean_memory():
    """ Decorator to clean memory before and after a function call
    """
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            clean_memory_fn()
            result = original_function(*args, **kwargs)
            clean_memory_fn()
            return result
        return wrapper
    return decorator
