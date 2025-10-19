"""
uv run src/infer.py \
  --checkpoint output/best_model.pt \
  --vocab output/phoneme_vocab.json \
  --text "שלום עולם"
"""
import torch
from transformers import AutoTokenizer
import argparse
import json
from pathlib import Path

from model import BertPhonemizer
from dataset import PhonemeVocabulary


def load_model(
    checkpoint_path: str,
    vocab_path: str,
    bert_model: str = 'dicta-il/dictabert',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Load trained model and vocabulary."""
    
    # Load phoneme vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    phoneme_vocab = PhonemeVocabulary(
        [p for p in vocab_data['phonemes'] if p not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']]
    )
    
    # Create model
    model = BertPhonemizer(
        bert_model_name=bert_model,
        phoneme_vocab_size=len(phoneme_vocab),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, phoneme_vocab


def phonemize(
    text: str,
    model: BertPhonemizer,
    tokenizer: AutoTokenizer,
    phoneme_vocab: PhonemeVocabulary,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
):
    """
    Convert text to phonemes using the trained model.
    """
    # Tokenize input
    encoding = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Generate phonemes
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bos_token_id=phoneme_vocab.bos_token_id,
            eos_token_id=phoneme_vocab.eos_token_id,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode phonemes
    phonemes = phoneme_vocab.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
    
    return phonemes


def main():
    parser = argparse.ArgumentParser(description='Phonemize text using trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to phoneme vocabulary JSON')
    parser.add_argument('--bert_model', type=str, default='dicta-il/dictabert', help='BERT model name')
    parser.add_argument('--text', type=str, help='Text to phonemize (optional, will use interactive mode if not provided)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}")
    print(f"Using device: {args.device}")
    
    # Load model and vocabulary
    model, phoneme_vocab = load_model(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        bert_model=args.bert_model,
        device=args.device,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    print("Model loaded successfully!\n")
    
    if args.text:
        # Single inference
        print(f"Text: {args.text}")
        phonemes = phonemize(
            text=args.text,
            model=model,
            tokenizer=tokenizer,
            phoneme_vocab=phoneme_vocab,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"Phonemes: {phonemes}")
    else:
        # Interactive mode
        print("Interactive mode - Enter text to phonemize (Ctrl+C to exit)")
        print("-" * 60)
        try:
            while True:
                text = input("\nText: ").strip()
                if not text:
                    continue
                
                phonemes = phonemize(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    phoneme_vocab=phoneme_vocab,
                    device=args.device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(f"Phonemes: {phonemes}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")


if __name__ == '__main__':
    main()