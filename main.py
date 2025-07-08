# main.py
import argparse
import sys
from config import config
from train import train
from generate import generate_summary, load_model_for_generation
from data.data_loader import Vocabulary # This is a simplification

def main():
    parser = argparse.ArgumentParser(description="AMR Abstractive Summarization Framework")
    parser.add_argument('mode', choices=['train', 'generate'], 
                        help="Mode to run the script in: 'train' or 'generate'")
    
    parser.add_argument('--model', type=str, default=config.MODEL_TYPE,
                        choices=['AS2SP', 'TRCE', 'PETR', 'RL'],
                        help='The type of model to train or use for generation.')
    
    parser.add_argument('--text', type=str,
                        help='The input text to summarize (only in generate mode).')
    
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained model file (only in generate mode).')

    args = parser.parse_args()

    # --- Update config based on arguments ---
    config.MODEL_TYPE = args.model
    
    if args.mode == 'train':
        print(f"Mode: Training")
        print(f"Model: {config.MODEL_TYPE}")
        print(f"Graph Construction: {config.GRAPH_CONSTRUCTION}")
        print(f"Graph Transformation: {config.GRAPH_TRANSFORMATION}")
        train()
        
    elif args.mode == 'generate':
        print(f"Mode: Generation")
        if not args.text or not args.model_path:
            print("Error: --text and --model_path are required for generate mode.", file=sys.stderr)
            sys.exit(1)
            
        # In a real application, vocab would be saved with the model
        # For now, we create a dummy one.
        print("Loading vocabulary (dummy)...")
        vocab = Vocabulary(config.VOCAB_SIZE)
        
        print(f"Loading model from {args.model_path}...")
        model = load_model_for_generation(args.model_path, vocab)
        
        print(f"Generating summary for: '{args.text}'")
        summary = generate_summary(model, args.text, vocab)
        print("\n--- Generated Summary ---")
        print(summary)
        print("------------------------")

if __name__ == '__main__':
    main()
