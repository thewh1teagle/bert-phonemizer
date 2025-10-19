"""
uv run src/train.py \
  --train_tsv data/train.tsv \
  --val_tsv data/val.tsv \
  --batch_size 32 \
  --num_epochs 10
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from model import BertPhonemizer
from dataset import TextPhonemeDataset, build_phoneme_vocab_from_file


def train(
    train_tsv: str,
    val_tsv: str = None,
    bert_model: str = 'dicta-il/dictabert',
    output_dir: str = 'output',
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    max_text_length: int = 128,
    max_phoneme_length: int = 256,
    decoder_layers: int = 6,
    decoder_heads: int = 8,
    decoder_dim: int = 768,
    dropout: float = 0.1,
    gradient_accumulation_steps: int = 1,
    save_every: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train the BERT-to-phoneme encoder-decoder model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Training on device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {bert_model}")
    text_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    
    # Build phoneme vocabulary from training data
    print(f"Building phoneme vocabulary from {train_tsv}")
    phoneme_vocab = build_phoneme_vocab_from_file(train_tsv)
    print(f"Phoneme vocabulary size: {len(phoneme_vocab)}")
    
    # Save vocabulary
    vocab_path = output_dir / 'phoneme_vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'phonemes': phoneme_vocab.phonemes,
            'phoneme_to_id': phoneme_vocab.phoneme_to_id,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved phoneme vocabulary to {vocab_path}")
    
    # Create datasets
    print(f"Loading training data from {train_tsv}")
    train_dataset = TextPhonemeDataset(
        tsv_file=train_tsv,
        text_tokenizer=text_tokenizer,
        phoneme_vocab=phoneme_vocab,
        max_text_length=max_text_length,
        max_phoneme_length=max_phoneme_length,
    )
    print(f"Training samples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
    )
    
    val_loader = None
    if val_tsv:
        print(f"Loading validation data from {val_tsv}")
        val_dataset = TextPhonemeDataset(
            tsv_file=val_tsv,
            text_tokenizer=text_tokenizer,
            phoneme_vocab=phoneme_vocab,
            max_text_length=max_text_length,
            max_phoneme_length=max_phoneme_length,
        )
        print(f"Validation samples: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device == 'cuda' else False,
        )
    
    # Create model
    print("Creating model...")
    model = BertPhonemizer(
        bert_model_name=bert_model,
        phoneme_vocab_size=len(phoneme_vocab),
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        decoder_dim=decoder_dim,
        max_phoneme_length=max_phoneme_length,
        dropout=dropout,
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    
    # Linear warmup then constant LR
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=phoneme_vocab.pad_token_id)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            phoneme_ids = batch['phoneme_ids'].to(device)
            phoneme_mask = batch['phoneme_mask'].to(device)
            
            # Forward pass (teacher forcing)
            # Shift phoneme_ids: input is [BOS, x1, x2, ...], target is [x1, x2, ..., EOS]
            decoder_input = phoneme_ids[:, :-1]
            decoder_input_mask = phoneme_mask[:, :-1]
            target = phoneme_ids[:, 1:]
            
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phoneme_ids=decoder_input,
                phoneme_mask=decoder_input_mask,
            )
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, len(phoneme_vocab)),
                target.reshape(-1)
            )
            
            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    phoneme_ids = batch['phoneme_ids'].to(device)
                    phoneme_mask = batch['phoneme_mask'].to(device)
                    
                    decoder_input = phoneme_ids[:, :-1]
                    decoder_input_mask = phoneme_mask[:, :-1]
                    target = phoneme_ids[:, 1:]
                    
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        phoneme_ids=decoder_input,
                        phoneme_mask=decoder_input_mask,
                    )
                    
                    loss = criterion(
                        logits.reshape(-1, len(phoneme_vocab)),
                        target.reshape(-1)
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} - Average validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = output_dir / 'best_model.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                }, best_model_path)
                print(f"Saved best model to {best_model_path}")
        
        # Save epoch checkpoint
        epoch_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'step': global_step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_path)
    
    print("\nTraining complete!")
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'step': global_step,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT Phonemizer')
    parser.add_argument('--train_tsv', type=str, required=True, help='Path to training TSV file')
    parser.add_argument('--val_tsv', type=str, default=None, help='Path to validation TSV file')
    parser.add_argument('--bert_model', type=str, default='dicta-il/dictabert', help='BERT model name')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--decoder_heads', type=int, default=8, help='Number of decoder attention heads')
    parser.add_argument('--decoder_dim', type=int, default=768, help='Decoder hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    train(
        train_tsv=args.train_tsv,
        val_tsv=args.val_tsv,
        bert_model=args.bert_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dim=args.decoder_dim,
        dropout=args.dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
    )