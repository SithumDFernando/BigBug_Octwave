import nbformat
import sys

def build_notebook(md_path, out_nb_path):
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {md_path}")
        sys.exit(1)

    nb = nbformat.v4.new_notebook()
    
    current_mode = 'markdown'
    current_block = []

    for line in lines:
        if line.strip() == '```python':
            if current_block and current_mode == 'markdown':
                nb.cells.append(nbformat.v4.new_markdown_cell(''.join(current_block).strip()))
                current_block = []
            current_mode = 'code'
        elif line.strip() == '```' and current_mode == 'code':
            if current_block:
                nb.cells.append(nbformat.v4.new_code_cell(''.join(current_block).strip('\n')))
                current_block = []
            current_mode = 'markdown'
        else:
            current_block.append(line)
            
    # handle last block
    if current_block:
        if current_mode == 'markdown':
            content = ''.join(current_block).strip()
            if content:
                nb.cells.append(nbformat.v4.new_markdown_cell(content))
        elif current_mode == 'code':
            content = ''.join(current_block).strip('\n')
            if content:
                nb.cells.append(nbformat.v4.new_code_cell(content))

    nbformat.write(nb, out_nb_path)
    print(f"Successfully generated {out_nb_path}")

if __name__ == "__main__":
    build_notebook('docs/notebook/notebook_reference.md', 'notebooks/Submissionnotebook.ipynb')
