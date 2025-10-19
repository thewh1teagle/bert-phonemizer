import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import pandas as pd


class PhonemeVocabulary:
    """Vocabulary for phoneme sequences."""
    
    def __init__(self, phonemes: List[str]):
        # Special tokens
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
        # Build vocabulary
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.phonemes = special_tokens + sorted(list(set(phonemes)))
        
        # Create mappings
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
        # Special token IDs
        self.pad_token_id = self.phoneme_to_id[self.pad_token]
        self.bos_token_id = self.phoneme_to_id[self.bos_token]
        self.eos_token_id = self.phoneme_to_id[self.eos_token]
        self.unk_token_id = self.phoneme_to_id[self.unk_token]
        
    def __len__(self):
        return len(self.phonemes)
    
    def encode(self, phoneme_str: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a phoneme string into IDs.
        Assumes phonemes are space-separated.
        """
        phonemes = phoneme_str.strip().split()
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for phoneme in phonemes:
            ids.append(self.phoneme_to_id.get(phoneme, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs back to phoneme string."""
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        
        phonemes = []
        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            phonemes.append(self.id_to_phoneme.get(id_, self.unk_token))
        
        return ' '.join(phonemes)


class TextPhonemeDataset(Dataset):
    """
    Dataset for text-to-phoneme mapping.
    Reads from TSV file with format: text\tphonemes
    """
    
    def __init__(
        self,
        tsv_file: str,
        text_tokenizer: PreTrainedTokenizer,
        phoneme_vocab: PhonemeVocabulary,
        max_text_length: int = 128,
        max_phoneme_length: int = 256,
    ):
        self.text_tokenizer = text_tokenizer
        self.phoneme_vocab = phoneme_vocab
        self.max_text_length = max_text_length
        self.max_phoneme_length = max_phoneme_length
        
        # Load data from TSV using pandas (no header)
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['text', 'phonemes'])
        self.data = df.to_dict('records')
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {tsv_file}. Expected TSV with text and phonemes columns.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize text
        text_encoding = self.text_tokenizer(
            item['text'],
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode phonemes
        phoneme_ids = self.phoneme_vocab.encode(item['phonemes'], add_special_tokens=True)
        
        # Truncate if too long
        if len(phoneme_ids) > self.max_phoneme_length:
            phoneme_ids = phoneme_ids[:self.max_phoneme_length-1] + [self.phoneme_vocab.eos_token_id]
        
        # Pad phonemes
        phoneme_mask = [1] * len(phoneme_ids)
        while len(phoneme_ids) < self.max_phoneme_length:
            phoneme_ids.append(self.phoneme_vocab.pad_token_id)
            phoneme_mask.append(0)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
            'phoneme_mask': torch.tensor(phoneme_mask, dtype=torch.long),
            'text': item['text'],
            'phonemes': item['phonemes']
        }


def build_phoneme_vocab_from_file(tsv_file: str) -> PhonemeVocabulary:
    """
    Build phoneme vocabulary from TSV file by collecting all unique phonemes.
    Assumes phonemes are space-separated and no header row.
    """
    # Read data with pandas (no header)
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['text', 'phonemes'])
    
    # Collect all unique phonemes
    all_phonemes = set()
    for phonemes_str in df['phonemes']:
        phonemes = phonemes_str.strip().split()
        all_phonemes.update(phonemes)
    
    return PhonemeVocabulary(list(all_phonemes))

