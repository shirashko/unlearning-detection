import json
import random

# Setting a seed for academic reproducibility
random.seed(42)

def get_random_sample(data_list, n=250):
    """Returns a random sample of n elements or the full list if shorter."""
    if not data_list:
        return []
    if len(data_list) <= n:
        return data_list
    return random.sample(data_list, n)

# 1. Load the original data
# Assuming 'data.json' contains the keys: multiplication_symbolic, multiplication_riddle, 
# division_symbolic, division_riddle, and english.
with open('data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 2. Extract and Group Concepts
# Multiplication Group
mult_list = raw_data.get("multiplication_symbolic", []) + raw_data.get("multiplication_riddle", [])

# Division Group
div_list = raw_data.get("division_symbolic", []) + raw_data.get("division_riddle", [])

# Neutral Group
neutral_list = raw_data.get("english", [])

# 3. Create Sampled Data for Multiplication
mult_target_data = {
    "target_concept": get_random_sample(mult_list, 250),
    "neutral": get_random_sample(neutral_list, 250)
}

# 4. Create Sampled Data for Division
div_target_data = {
    "target_concept": get_random_sample(div_list, 250),
    "neutral": get_random_sample(neutral_list, 250)
}

# 5. Save the separate JSON files
with open('mult_target_data.json', 'w', encoding='utf-8') as f:
    json.dump(mult_target_data, f, indent=4, ensure_ascii=False)

with open('div_target_data.json', 'w', encoding='utf-8') as f:
    json.dump(div_target_data, f, indent=4, ensure_ascii=False)

print("Files 'mult_target_data.json' and 'div_target_data.json' have been created.")