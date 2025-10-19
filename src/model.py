import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class BertPhonemizer(nn.Module):
    """
    Encoder-Decoder model for phonemization.
    - Encoder: Frozen BERT model
    - Decoder: Transformer decoder with causal attention
    """
    
    def __init__(
        self, 
        bert_model_name: str = 'dicta-il/dictabert',
        phoneme_vocab_size: int = 64,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_dim: int = 768,
        decoder_ff_dim: int = 2048,
        max_phoneme_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Frozen BERT encoder
        self.encoder = AutoModel.from_pretrained(bert_model_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Phoneme vocabulary
        self.phoneme_vocab_size = phoneme_vocab_size
        self.max_phoneme_length = max_phoneme_length
        
        # Phoneme embeddings for decoder
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, decoder_dim)
        self.positional_encoding = nn.Embedding(max_phoneme_length, decoder_dim)
        
        # Project encoder hidden states to decoder dimension if needed
        if self.encoder_hidden_size != decoder_dim:
            self.encoder_projection = nn.Linear(self.encoder_hidden_size, decoder_dim)
        else:
            self.encoder_projection = nn.Identity()
        
        # Transformer decoder with causal attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
        # Output projection to phoneme vocabulary
        self.output_projection = nn.Linear(decoder_dim, phoneme_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [batch, src_len] - Text token IDs from BERT tokenizer
            attention_mask: [batch, src_len] - Attention mask for text
            phoneme_ids: [batch, tgt_len] - Phoneme IDs (for teacher forcing during training)
            phoneme_mask: [batch, tgt_len] - Mask for phoneme padding
            
        Returns:
            logits: [batch, tgt_len, phoneme_vocab_size] - Predictions for each phoneme position
        """
        batch_size = input_ids.shape[0]
        
        # Encode text with frozen BERT
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch, src_len, hidden]
        
        # Project encoder outputs to decoder dimension
        encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
        
        # For training: use teacher forcing
        if phoneme_ids is not None:
            tgt_len = phoneme_ids.shape[1]
            
            # Embed phonemes
            phoneme_embeds = self.phoneme_embedding(phoneme_ids)  # [batch, tgt_len, decoder_dim]
            
            # Add positional encoding
            positions = torch.arange(tgt_len, device=phoneme_ids.device).unsqueeze(0)
            pos_embeds = self.positional_encoding(positions)
            phoneme_embeds = phoneme_embeds + pos_embeds
            phoneme_embeds = self.dropout(phoneme_embeds)
            
            # Create causal mask for decoder (triangular mask)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=phoneme_ids.device
            )
            
            # Create key padding mask for encoder (inverted attention_mask)
            # True values are ignored in attention
            memory_key_padding_mask = ~attention_mask.bool()
            
            # Create key padding mask for decoder target
            tgt_key_padding_mask = None
            if phoneme_mask is not None:
                tgt_key_padding_mask = ~phoneme_mask.bool()
            
            # Decode
            decoder_output = self.decoder(
                tgt=phoneme_embeds,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(decoder_output)
            
            return logits
        
        # For inference: autoregressive generation (handled in generate method)
        else:
            raise ValueError("Use the generate() method for inference without phoneme_ids")
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        Autoregressive generation of phoneme sequences.
        
        Args:
            input_ids: [batch, src_len] - Text token IDs
            attention_mask: [batch, src_len] - Attention mask for text
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated_ids: [batch, generated_len] - Generated phoneme IDs
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if max_length is None:
            max_length = self.max_phoneme_length
        
        # Encode text with frozen BERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden_states = self.encoder_projection(encoder_outputs.last_hidden_state)
        
        # Initialize decoder input with BOS token
        generated_ids = torch.full(
            (batch_size, 1), 
            bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Track which sequences are done
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate autoregressively
        for step in range(max_length - 1):
            tgt_len = generated_ids.shape[1]
            
            # Embed current sequence
            phoneme_embeds = self.phoneme_embedding(generated_ids)
            positions = torch.arange(tgt_len, device=device).unsqueeze(0)
            pos_embeds = self.positional_encoding(positions)
            phoneme_embeds = phoneme_embeds + pos_embeds
            
            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )
            
            # Create encoder padding mask
            memory_key_padding_mask = ~attention_mask.bool()
            
            # Decode
            decoder_output = self.decoder(
                tgt=phoneme_embeds,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])  # [batch, vocab_size]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # For finished sequences, keep predicting EOS
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_token_id),
                next_token
            )
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Update finished status
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            
            # Stop if all sequences are done
            if finished.all():
                break
        
        return generated_ids

