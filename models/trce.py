# models/trce.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TRCEModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.dec_embed = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_texts, trg_ids):
        # Tokenize source texts with BERT tokenizer
        bert_inputs = self.bert_tokenizer(
            src_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(trg_ids.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(**bert_inputs)
            src_emb = bert_output.last_hidden_state

        # Get target embeddings and create mask
        trg_emb = self.dropout(self.dec_embed(trg_ids))
        trg_mask = self.generate_square_subsequent_mask(trg_ids.size(1)).to(trg_ids.device)
        
        # Get padding masks
        src_key_padding_mask = (bert_inputs.input_ids == self.bert_tokenizer.pad_token_id)
        tgt_key_padding_mask = (trg_ids == 0) # Assuming 0 is pad token

        # Pass through transformer
        output = self.transformer(
            src_emb, 
            trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
