import json
import sys

for nb_name in ['01_EDA', '02_Data_Preprocessing', '03_Model_Training', '04_Evaluation_and_Prediction']:
    path = f'notebooks/{nb_name}.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    sep = '=' * 80
    lines = []
    lines.append(f'\n{sep}')
    lines.append(f'NOTEBOOK: {nb_name}')
    lines.append(f'Total cells: {len(nb["cells"])}')
    lines.append(sep)
    
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        src = ''.join(cell.get('source', []))
        if ct == 'markdown':
            lines.append(f'\n--- [MARKDOWN Cell {i}] ---')
            lines.append(src)
    
    with open('docs/notebook/notebook_markdown_dump.md', 'a', encoding='utf-8') as out:
        out.write('\n'.join(lines))
