import re
def parse_output(output: str):
    matches = re.finditer(r'([^<]*)<([^\s>]*)>', output)
    items = []
    crakets = []
    for match in matches:
        items.append(match.group(1).strip())
        crakets.append(match.group(2).strip())
    return items ,crakets

def extract_yes_no_and_map(text):
    # Convert the input text to lowercase for case-insensitive matching
    text = text.lower()

    # Define regular expressions for yes/no matching
    yes_patterns = [r'\byes\b', r'\btrue\b']
    no_patterns = [r'\bno\b', r'\bfalse\b']

    # Check for "0"
    if text == "0":
        return "0"

    # Check for "1"
    if text == "1":
        return "1"

    # Check for yes
    for pattern in yes_patterns:
        if re.search(pattern, text):
            return "1"

    # Check for no
    for pattern in no_patterns:
        if re.search(pattern, text):
            return "0"

    # Return 2 if neither yes nor no is found
    return "2"

def eval_fv_match(pred_list, gold_list):
        acc = 0.0
        for pred, gold in zip(pred_list, gold_list):
            pred, gold = extract_yes_no_and_map(pred), extract_yes_no_and_map(gold)
            if pred == gold:
                acc += 1
        acc = acc / len(pred_list)
        return acc
    