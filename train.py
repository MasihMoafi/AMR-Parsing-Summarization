# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
from config import config
from data.data_loader import get_dataloader, Vocabulary
from models.as2sp import AS2SP
from models.trce import TRCEModel
from models.petr import PETRModel
from models.rl import self_critical_loss

def get_model(vocab_size):
    if config.MODEL_TYPE == 'AS2SP':
        return AS2SP(vocab_size, config.EMBED_SIZE, config.HIDDEN_SIZE, config.ENC_HIDDEN_SIZE, config.DROPOUT)
    elif config.MODEL_TYPE == 'TRCE':
        return TRCEModel(vocab_size, config.TRANSFORMER_D_MODEL, config.TRANSFORMER_NHEAD, 
                         config.TRANSFORMER_ENCODER_LAYERS, config.TRANSFORMER_DECODER_LAYERS, 
                         config.TRANSFORMER_DIM_FEEDFORWARD, config.DROPOUT)
    elif config.MODEL_TYPE == 'PETR':
        return PETRModel(vocab_size, config.TRANSFORMER_D_MODEL, config.TRANSFORMER_NHEAD,
                         config.TRANSFORMER_ENCODER_LAYERS, config.TRANSFORMER_DECODER_LAYERS,
                         config.TRANSFORMER_DIM_FEEDFORWARD, config.DROPOUT)
    elif config.MODEL_TYPE == 'RL':
        # RL uses AS2SP as its base model
        return AS2SP(vocab_size, config.EMBED_SIZE, config.HIDDEN_SIZE, config.ENC_HIDDEN_SIZE, config.DROPOUT)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

def train():
    device = torch.device(config.DEVICE)
    
    # -- Data Loading --
    vocab = Vocabulary(config.VOCAB_SIZE)
    train_loader = get_dataloader(config.TRAIN_CSV, config.BATCH_SIZE, vocab, 
                                  config.GRAPH_CONSTRUCTION, config.GRAPH_TRANSFORMATION, is_train=True)
    # val_loader = get_dataloader(config.VALIDATION_CSV, config.BATCH_SIZE, vocab, 
    #                             config.GRAPH_CONSTRUCTION, config.GRAPH_TRANSFORMATION, is_train=False)

    # -- Model & Optimizer --
    model = get_model(len(vocab.word2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"--- Starting Training for {config.MODEL_TYPE} ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for src_data, trg_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            trg_ids = trg_ids.to(device)
            
            optimizer.zero_grad()
            
            if config.MODEL_TYPE == 'RL':
                loss = self_critical_loss(model, src_data, trg_ids, vocab)
            else:
                # Standard training for AS2SP, TRCE, PETR
                # Note: src_data is a list of strings for transformer models
                # and will need to be converted to IDs for AS2SP. This is a simplification.
                
                # For now, we assume the dataloader gives strings and models handle tokenization
                outputs = model(src_data, trg_ids[:, :-1])
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)), 
                    trg_ids[:, 1:].reshape(-1), 
                    ignore_index=vocab.word2idx["<pad>"]
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Validation loop would go here

    print("--- Training Completed ---")
    # Saving model logic would go here

if __name__ == '__main__':
    train()
