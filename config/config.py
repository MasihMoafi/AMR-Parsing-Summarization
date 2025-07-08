# config.py

# -- Path Settings --
# It's recommended to use absolute paths
DATA_PATH = '/home/masih/Desktop/AMR/data/'
TRAIN_CSV = DATA_PATH + 'train.csv'
VALIDATION_CSV = DATA_PATH + 'validation.csv'
TEST_CSV = DATA_PATH + 'test.csv'
MODEL_PATH = '/home/masih/Desktop/AMR/model_parse_xfm_bart_large-v0_1_0' # Path to the extracted AMR parsing model
SAVE_PATH = '/home/masih/Desktop/AMR/saved_models/' # Directory to save trained models

# -- Model Selection --
# Choose from: 'AS2SP', 'TRCE', 'PETR', 'RL'
MODEL_TYPE = 'AS2SP'

# -- Data & Graph Settings --
# Graph Construction: 'sequence' or 'combination'
GRAPH_CONSTRUCTION = 'sequence'
# Graph Transformation: 'OAMR', 'OAMRWS', 'SAMR', 'SAMRWS'
GRAPH_TRANSFORMATION = 'SAMRWS'


# -- Model Hyperparameters --
VOCAB_SIZE = 50000
EMBED_SIZE = 128
HIDDEN_SIZE = 256
ENC_HIDDEN_SIZE = 128 # For AS2SP

# Transformer settings (for TRCE, PETR)
TRANSFORMER_D_MODEL = 768
TRANSFORMER_NHEAD = 8
TRANSFORMER_ENCODER_LAYERS = 6
TRANSFORMER_DECODER_LAYERS = 6
TRANSFORMER_DIM_FEEDFORWARD = 2048

# -- Training Settings --
DEVICE = 'cuda' # 'cuda' or 'cpu'
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GRADIENT_CLIP = 1.0
DROPOUT = 0.2

# RL specific
RL_LEARNING_RATE = 1e-4

# Transformer specific
TR_LEARNING_RATE = 0.05
TR_WARMUP_STEPS = 10000
PETR_ENC_LEARNING_RATE = 0.002
PETR_DEC_LEARNING_RATE = 0.1
PETR_ENC_WARMUP_STEPS = 20000
PETR_DEC_WARMUP_STEPS = 10000

# -- Generation Settings --
MAX_SUMMARY_LEN = 100
BEAM_WIDTH = 5
