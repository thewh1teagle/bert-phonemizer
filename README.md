# BERT Phonemizer

An encoder-decoder model for Hebrew text-to-phoneme conversion using a frozen BERT encoder and trainable transformer decoder with causal attention.

## Architecture

- **Encoder**: Frozen DictaBERT model (dicta-il/dictabert)
- **Decoder**: Transformer decoder with causal attention (trainable)
- **Task**: Text → Phoneme sequence generation

The BERT encoder is frozen to leverage pre-trained Hebrew language understanding, while the decoder learns to generate phoneme sequences autoregressively.

## Installation

```bash
pip install torch transformers tqdm
```

Or if using `uv`:
```bash
uv sync
```

## Data Format

Your training data should be a TSV (tab-separated values) file with two columns:
- `text`: The input text (Hebrew)
- `phonemes`: Space-separated phoneme sequence

Example (`data.tsv`):
```tsv
text	phonemes
שלום	ʃ a l o m
בוקר טוב	b o k e ʁ t o v
אני אוהב מוזיקה	a n i o h e v m u z i k a
```

## Training

Train the model with your TSV data:

```bash
python src/train.py \
  --train_tsv data/train.tsv \
  --val_tsv data/val.tsv \
  --output_dir output \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --decoder_layers 6 \
  --decoder_heads 8 \
  --decoder_dim 768
```

### Training Arguments

- `--train_tsv`: Path to training TSV file (required)
- `--val_tsv`: Path to validation TSV file (optional)
- `--bert_model`: BERT model name (default: `dicta-il/dictabert`)
- `--output_dir`: Output directory for checkpoints (default: `output`)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Learning rate warmup steps (default: 1000)
- `--decoder_layers`: Number of decoder layers (default: 6)
- `--decoder_heads`: Number of decoder attention heads (default: 8)
- `--decoder_dim`: Decoder hidden dimension (default: 768)
- `--dropout`: Dropout rate (default: 0.1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--save_every`: Save checkpoint every N steps (default: 1000)

### Output

The training script will save:
- `phoneme_vocab.json`: Phoneme vocabulary
- `checkpoint_step_*.pt`: Checkpoints every N steps
- `checkpoint_epoch_*.pt`: Checkpoints after each epoch
- `best_model.pt`: Best model based on validation loss
- `final_model.pt`: Final model after training

## Inference

Use the trained model to convert text to phonemes:

### Single Text
```bash
python src/infer.py \
  --checkpoint output/best_model.pt \
  --vocab output/phoneme_vocab.json \
  --text "שלום עולם"
```

### Interactive Mode
```bash
python src/infer.py \
  --checkpoint output/best_model.pt \
  --vocab output/phoneme_vocab.json
```

### Inference Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--vocab`: Path to phoneme vocabulary JSON (required)
- `--bert_model`: BERT model name (default: `dicta-il/dictabert`)
- `--text`: Text to phonemize (optional, interactive mode if not provided)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_k`: Top-k sampling (optional)
- `--top_p`: Nucleus sampling (optional)
- `--device`: Device to use (default: auto-detect)

## How It Works

1. **Encoding**: Text is tokenized with BERT tokenizer and encoded by frozen BERT
2. **Decoding**: Transformer decoder with causal attention generates phonemes autoregressively
3. **Training**: Teacher forcing with cross-entropy loss (ignoring padding tokens)
4. **Inference**: Autoregressive generation with optional sampling strategies

### Key Features

- ✅ Frozen BERT encoder (no gradient updates)
- ✅ Causal attention in decoder
- ✅ Teacher forcing during training
- ✅ Autoregressive generation during inference
- ✅ Gradient accumulation support
- ✅ Learning rate warmup
- ✅ Checkpoint saving and resumption
- ✅ Validation loss tracking
- ✅ Flexible sampling strategies (temperature, top-k, top-p)

## Example Usage

```python
from model import BertPhonemizer
from dataset import PhonemeVocabulary
from transformers import AutoTokenizer
import torch
import json

# Load vocabulary
with open('output/phoneme_vocab.json', 'r') as f:
    vocab_data = json.load(f)
phoneme_vocab = PhonemeVocabulary(
    [p for p in vocab_data['phonemes'] if p not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']]
)

# Load model
model = BertPhonemizer(
    bert_model_name='dicta-il/dictabert',
    phoneme_vocab_size=len(phoneme_vocab),
)
checkpoint = torch.load('output/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')

# Phonemize text
text = "שלום עולם"
encoding = tokenizer(text, return_tensors='pt')
generated_ids = model.generate(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask'],
    bos_token_id=phoneme_vocab.bos_token_id,
    eos_token_id=phoneme_vocab.eos_token_id,
)
phonemes = phoneme_vocab.decode(generated_ids[0].tolist(), skip_special_tokens=True)
print(f"Text: {text}")
print(f"Phonemes: {phonemes}")
```

## License

MIT