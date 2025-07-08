# models/as2sp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AS2SP(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size, dropout):
        super().__init__()
        self.enc_embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, enc_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        
        self.dec_embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_proj = nn.Linear(enc_hidden_size * 2, hidden_size)
        self.cell_proj = nn.Linear(enc_hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        
        # Attention
        self.W_h = nn.Linear(enc_hidden_size * 2, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        # Pointer-Generator
        self.p_gen_linear = nn.Linear(enc_hidden_size * 2 + hidden_size + embed_size, 1)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_graph, trg_text):
        # This is a simplified forward pass. A full implementation would handle
        # padding, attention, and pointer-generator logic more explicitly.
        enc_embedded = self.dropout(self.enc_embed(src_graph))
        enc_out, (h_n, c_n) = self.encoder(enc_embedded)

        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=-1)
        c_n_cat = torch.cat([c_n[0], c_n[1]], dim=-1)

        decoder_hidden = self.hidden_proj(h_n_cat).unsqueeze(0)
        decoder_cell = self.cell_proj(c_n_cat).unsqueeze(0)

        dec_embedded = self.dropout(self.dec_embed(trg_text))
        dec_out, (dec_h, dec_c) = self.decoder(dec_embedded, (decoder_hidden, decoder_cell))
        
        # In a real implementation, attention and pointer-generator logic would go here
        
        predictions = self.fc(dec_out)
        return predictions
