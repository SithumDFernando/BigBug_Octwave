import nbformat
import os

notebooks = [
    'notebooks/01_EDA.ipynb',
    'notebooks/02_Data_Preprocessing.ipynb',
    'notebooks/04_Evaluation_and_Prediction.ipynb',
]

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"--- {nb_path} NOT FOUND ---")
        continue
    nb = nbformat.read(nb_path, as_version=4)
    print(f"\n=== {nb_path} ({len(nb.cells)} cells) ===")
    for i, c in enumerate(nb.cells):
        outputs = c.get('outputs', [])
        has_image = any('image/png' in str(o.get('data', {})) for o in outputs)
        src = c.source[:120].replace('\n', ' | ')
        if has_image:
            print(f"  [IMG] Cell {i} ({c.cell_type}): {src}")
        elif c.cell_type == 'code' and outputs:
            print(f"  [OUT] Cell {i} ({c.cell_type}): {src}")
