import os

if not os.path.exists("data/per_query_qrels"):
    os.mkdir("data/per_query_qrels")

def split_file_by_number(input_file, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold lines for each number
    number_lines = {}

    with open(input_file, 'r') as file:
        for line in file:
            # Extract the starting number
            if line.strip():  # Skip empty lines
                parts = line.strip().split(maxsplit=1)
                number = parts[0]  # First part is the number
                content = parts[1] if len(parts) > 1 else ''

                # Add the line to the corresponding number
                if number not in number_lines:
                    number_lines[number] = []
                number_lines[number].append(line)

    # Write lines to separate files
    for number, lines in number_lines.items():
        output_file = os.path.join(output_dir, f"{number}.txt")
        with open(output_file, 'w') as out_file:
            out_file.writelines(lines)

# Usage
input_file = "data/robust03_qrels.txt"
output_dir = "data/per_query_qrels"
split_file_by_number(input_file, output_dir)