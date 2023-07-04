import os
import re
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize
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
SUBCONTEXT_KEYS = [  # TODO (?) ASSOCIATE EVERY KEY TO A CONTEXT KEY (?)
    'infants?', 'allowed for infants?', 'women', 'allowed for women',
    'allowed for pregnant women', 'life expectancy', 'hematopoietic',
    'hematologic', 'hepatic', 'renal', 'cardiovascular', 'cardiac', 'pulmonary',
    'systemic', 'biologic therapy', 'chemotherapy', 'endocrine therapy',
    'radiotherapy', 'surgery', 'other', 'performance status', 'age', 'sex',
    'definitive local therapy', 'previous hormonal therapy or other treatments?',
    'other treatments?', 'previous hormonal therapy', 'in either eyes?',
    'in study eyes?', 'one of the following', 'body mass index', 'bmi',
    'eligible subtypes?', 'hormone receptor status', 'menopausal status',
    'at least 1 of the following factors',
]
NON_CONTEXT_KEYS = ['notes?', 'amended']  # not used for now
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
    
    def __iter__(self):
        for ct_path, ct_dict in self.dp:
            # Load protocol and make sure it corresponds to the file name
            protocol = ct_dict['FullStudy']['Study']['ProtocolSection']
            nct_id = protocol['IdentificationModule']['NCTId']
            assert nct_id == os.path.split(ct_path)[-1].strip('.json')
            
            # Check protocol belongs to the data and load criteria
            good_to_go, label, phases, conditions = self.check_protocol(protocol)
            if good_to_go:
                metadata = {
                    'ct_path': ct_path,
                    'label': label,
                    'phases': phases,
                    'conditions': conditions,
                }
                criteria_str = protocol['EligibilityModule']['EligibilityCriteria']
                yield metadata, criteria_str
                
    def check_protocol(self, protocol):
        """ Parse clinical trial protocol and make sure it can be used as a data
            sample for eligibility criteria representation learning 
        """
        # Check the status of the CT is either completed or terminated
        status = protocol['StatusModule']['OverallStatus']
        if status not in ['Completed', 'Terminated']:
            return False, None, None, None
        
        # Check that the study is interventional
        study_type = protocol['DesignModule']['StudyType']
        if study_type != 'Interventional':
            return False, None, None, None
        
        # Check the study is about a drug test
        interventions = protocol[
            'ArmsInterventionsModule']['InterventionList']['Intervention']
        intervention_types = [i['InterventionType'] for i in interventions]
        if 'Drug' not in intervention_types:
            return False, None, None, None
        
        # Check the study has defined phases, then record phases
        if 'PhaseList' not in protocol['DesignModule'].keys():
            return False, None, None, None
        phases = protocol['DesignModule']['PhaseList']['Phase']
        # if 'Not Applicable' in phases:
        #     return False, None, None, None
        
        # Check that the protocol has an eligibility criterion section
        if 'EligibilityCriteria' not in protocol['EligibilityModule'].keys():
            return False, None, None, None
        
        # Check that the protocol has a condition list
        if 'ConditionList' not in protocol['ConditionsModule']:
            return False, None, None, None
        conditions = protocol['ConditionsModule']['ConditionList']['Condition']
                      
        # Return that the protocol can be processed, status, and phase list
        return True, status, phases, conditions
    
    
@functional_datapipe('parse_criteria')
class CriteriaParser(IterDataPipe):
    """ Parse criteria raw paragraphs into a set of individual criteria, as well
        as other features and labels, such as medical context
    """
    def __init__(self, dp):
        super().__init__()
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
        parsed = self.contextualize_criteria_with_colons(parsed)
        parsed = self.post_process_criteria_by_semicolon(parsed)
        # parsed = self.post_process_criteria_by_comma(parsed)  # produced too many false positives
        
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
    def contextualize_criteria_with_colons(parsed_criteria: list[dict[str, str]],
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
                contextualized.append({'category': criterion['category'],
                                        'context': context,
                                        'subcontext': subcontext,
                                        'text': split.strip('\n\t :')})
            
            # # Small check in case previous category was different (ok?)
            # if category != prev_category:
            #     context, subcontext = '', ''
            #     prev_category = category
            
        # Return newly parsed set of criteria, with added context
        return contextualized
    
    @staticmethod
    def post_process_criteria_by_semicolon(parsed_criteria: list[dict[str, str]],
                                           ) -> list[dict[str, str]]:
        """ Simply split each criterion text by semi-colon (could try to fuse
            back split parts that are too short?)
        """
        post_parsed = []
        for criterion in parsed_criteria:
            splits = criterion['text'].split(';')
            post_parsed.extend([dict(criterion, text=s.strip()) for s in splits])
        return post_parsed
    
    @staticmethod
    def split_by_semicolon(s: str, placeholder='*') -> list[str]:
        """ Split a string by semicolons into big phrases, trying to avoid false
            positive cases, such as within parentheses or quotation marks
        """
        # Replace semicolons that are not big phrase separators by '*' characters
        regex = r'\([^)]*\)|\[[^\]]*\]|"[^"]*"|\*[^*]*\*|\'[^\']*\''
        replace_fn = lambda match: match.group(0).replace(';', placeholder)
        s = re.sub(regex, replace_fn, s)
        
        # Split by commas and put back commas that were replaced
        splits = s.split(';')
        return [split.replace(placeholder, ';') for split in splits]
    
        
@functional_datapipe('write_csv')
class CriteriaCSVWriter(IterDataPipe):
    """ Take the output of CriteriaParser (list of dictionaries) and transform
        it into a list of lists of strings, ready to be written to a csv file
    """
    def __init__(self, dp: IterDataPipe) -> None:
        super().__init__()
        self.dp = dp
        
    def __iter__(self) -> None:
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
            metadata['ct_path'] if i == 0 else '',
            metadata['complexity'] if i == 0 else '',
            metadata['label'],
            metadata['phases'],
            metadata['conditions'],
            c['category'],
            c['context'],
            c['subcontext'],
            c['text'].replace('≥', '> or equal to')
                     .replace('≤', '< or equal to')
                     .strip('- '),
        ] for i, c in enumerate(parsed_criteria)]
        