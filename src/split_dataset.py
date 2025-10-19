"""Split dataset into train and validation sets.

Usage:
uv run src/split_dataset.py \
    --input data/dataset.tsv \
    --train_file data/train.tsv \
    --val_file data/val.tsv \
    --val_ratio 0.01
"""

import argparse
import pandas as pd


def split_dataset(
    input_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.01,
):
    """
    Split dataset into train and validation sets.
    
    Args:
        input_file: Input TSV file
        train_file: Output training TSV file
        val_file: Output validation TSV file
        val_ratio: Ratio of validation data (default: 0.01 = 1%)
    """
    # Read data (no header)
    df = pd.read_csv(input_file, sep='\t', header=None, names=['text', 'phonemes'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    val_size = int(len(df) * val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]
    
    # Write to files (no header)
    train_df.to_csv(train_file, sep='\t', index=False, header=False)
    val_df.to_csv(val_file, sep='\t', index=False, header=False)
    
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Training samples: {len(train_df)} ({(1-val_ratio)*100:.1f}%)")
    print(f"✓ Validation samples: {len(val_df)} ({val_ratio*100:.1f}%)")
    print(f"✓ Training data saved to: {train_file}")
    print(f"✓ Validation data saved to: {val_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train and validation sets')
    parser.add_argument('--input', type=str, required=True, help='Input TSV file')
    parser.add_argument('--train_file', type=str, required=True, help='Output training TSV file')
    parser.add_argument('--val_file', type=str, required=True, help='Output validation TSV file')
    parser.add_argument('--val_ratio', type=float, default=0.01, help='Ratio of validation data (default: 0.01 = 1%)')
    args = parser.parse_args()
    split_dataset(args.input, args.train_file, args.val_file, args.val_ratio)