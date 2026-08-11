import nbformat
import os

def build_working_notebook(md_path, src_dir, out_nb_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    nb = nbformat.v4.new_notebook()
    
    current_md_block = []
    in_code_block = False

    def add_md():
        if current_md_block:
            content = ''.join(current_md_block).strip()
            if content:
                nb.cells.append(nbformat.v4.new_markdown_cell(content))
            current_md_block.clear()

    def add_src_code(relative_path):
        full_path = os.path.join(src_dir, relative_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
            nb.cells.append(nbformat.v4.new_code_cell(code))
            print(f"Injected {relative_path}")
        else:
            print(f"Warning: {relative_path} not found!")

    for line in lines:
        if line.strip() == '```python':
            in_code_block = True
            continue
        if in_code_block and line.strip() == '```':
            in_code_block = False
            continue
            
        if in_code_block:
            continue # skip pseudo-code

        # Injection points
        if line.startswith('## 6. Phase 5: Model Training'):
            add_md()
            add_src_code('data_processing/preprocess.py')
            
        elif line.startswith('## 7. Phase 6: Ensemble Construction'):
            add_md()
            add_src_code('modeling/train.py')
            
        elif line.startswith('## 8. Phase 7: Advanced Ensemble'):
            add_md()
            add_src_code('modeling/ensemble.py')
            
        elif line.startswith('## 9. Phase 8: Final Model Selection'):
            add_md()
            add_src_code('modeling/advanced_ensemble.py')

        current_md_block.append(line)
        
    add_md() # add any remaining markdown
    
    nbformat.write(nb, out_nb_path)
    print(f"Successfully generated {out_nb_path}")

if __name__ == "__main__":
    md_file = 'docs/notebook/notebook_reference.md'
    src_folder = 'src'
    out_nb = 'notebooks/Working_Submission.ipynb'
    build_working_notebook(md_file, src_folder, out_nb)
