import re
def parse_output(output: str):
    matches = re.finditer(r'([^<]*)<([^\s>]*)>', output)
    items = []
    crakets = []
    for match in matches:
        items.append(match.group(1).strip())
        crakets.append(match.group(2).strip())
    return items ,crakets
    