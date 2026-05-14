import json
from transformers import AutoTokenizer

# 1. Configuration
MODEL_ID = "google/gemma-2-2b"  # Use the model you are actually training/testing on
MAX_TOKENS = 256
INPUT_FILE = "div_mult_neutral_data.json"
OUTPUT_FILE = "div_mult_neutral_data_truncated_256.json"

def truncate_examples(data, tokenizer, max_length):
    truncated_data = {}
    
    for category, examples in data.items():
        print(f"Processing category: {category}...")
        new_examples = []
        
        for text in examples:
            # Tokenize the text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Check if truncation is needed
            if len(tokens) > max_length:
                # Truncate and decode back to string
                truncated_tokens = tokens[:max_length]
                truncated_text = tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)
                new_examples.append(truncated_text)
            else:
                new_examples.append(text)
                
        truncated_data[category] = new_examples
        
    return truncated_data

def main():
    # Load the tokenizer
    # Note: If prompted for login, ensure you are logged into Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load your JSON data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Process
    final_data = truncate_examples(raw_data, tokenizer, MAX_TOKENS)

    # Save the result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"\nSuccess! Truncated data saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()