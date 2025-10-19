"""
Prepare dataset by tokenizing concatenated phonemes into space-separated format.

Download data:
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
7z x knesset_phonemes_v1.txt.7z
head -n 1000 knesset_phonemes_v1.txt > data.tsv

Usage:
uv run src/prepare_dataset.py \
    --input data.tsv \
    --output data/dataset.tsv
"""

import argparse
import re
from pathlib import Path
from typing import List, Set
from tqdm import tqdm
import pandas as pd

from vocab import HEBREW_PHONEMES


def remove_diacritics(text: str) -> str:
    text = re.sub(r'[\u0590-\u05C7|]', '', text)
    return text


def tokenize_phonemes_longest_match(phoneme_str: str, known_phonemes: Set[str]) -> List[str]:
    """
    Tokenize a concatenated phoneme string using longest-match (greedy) algorithm.
    
    Args:
        phoneme_str: Concatenated phoneme string (e.g., "ʔavakˈeʃ")
        known_phonemes: Set of known phonemes including multi-character ones
    
    Returns:
        List of tokenized phonemes
    """
    # Sort phonemes by length (longest first) for greedy matching
    sorted_phonemes = sorted(known_phonemes, key=len, reverse=True)
    
    result = []
    i = 0
    
    while i < len(phoneme_str):
        matched = False
        
        # Try to match longest phoneme first
        for phoneme in sorted_phonemes:
            if phoneme_str[i:].startswith(phoneme):
                result.append(phoneme)
                i += len(phoneme)
                matched = True
                break
        
        if not matched:
            # Single character as fallback
            result.append(phoneme_str[i])
            i += 1
    
    return result


def process_dataset(
    input_file: str,
    output_file: str,
    known_phonemes: Set[str],
    skip_invalid: bool = True,
):
    """
    Process dataset by tokenizing concatenated phonemes.
    
    Args:
        input_file: Path to input file (tab-separated: text\tphonemes)
        output_file: Path to output TSV file
        known_phonemes: Set of known phonemes
        skip_invalid: Skip lines with invalid format
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read data with pandas (no header)
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, sep='\t', header=None, names=['text', 'phonemes'])
    
    print(f"Processing {len(df)} lines...")
    
    # Remove diacritics from text
    tqdm.pandas(desc="Removing diacritics")
    df['text'] = df['text'].progress_apply(remove_diacritics)
    
    # Tokenize phonemes
    tqdm.pandas(desc="Tokenizing phonemes")
    df['phonemes'] = df['phonemes'].progress_apply(
        lambda x: ' '.join(tokenize_phonemes_longest_match(x, known_phonemes))
    )
    
    # Write output (no header)
    df.to_csv(output_path, sep='\t', index=False, header=False)
    
    print(f"\n✓ Processed {len(df)} lines")
    print(f"✓ Output saved to: {output_path}")
    
    # Show some statistics
    print("\nSample outputs:")
    for i in range(min(3, len(df))):
        text = df.iloc[i]['text']
        phonemes = df.iloc[i]['phonemes']
        print(f"  Text: {text[:50]}...")
        print(f"  Phonemes: {phonemes[:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset by tokenizing concatenated phonemes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/prepare_dataset.py \\
      --input knesset_phonemes_v1.txt \\
      --output data/train.tsv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path (tab-separated: text\\tphonemes)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output TSV file path'
    )
    
    parser.add_argument(
        '--skip-invalid',
        action='store_true',
        default=True,
        help='Skip lines with invalid format (default: True)'
    )
    
    args = parser.parse_args()
    
    # Use known phonemes from vocab.py
    print(f"Using {len(HEBREW_PHONEMES)} known phonemes from vocab.py")
    multi_char = sorted([p for p in HEBREW_PHONEMES if len(p) > 1], key=len, reverse=True)
    print(f"Multi-character phonemes: {multi_char}")
    
    # Process dataset
    process_dataset(
        input_file=args.input,
        output_file=args.output,
        known_phonemes=HEBREW_PHONEMES,
        skip_invalid=args.skip_invalid,
    )


if __name__ == '__main__':
    main()