import nbformat
import json

notebooks = {
    'EDA': 'notebooks/01_EDA.ipynb',
    'Preprocess': 'notebooks/02_Data_Preprocessing.ipynb',
    'Evaluation': 'notebooks/04_Evaluation_and_Prediction.ipynb',
}

for label, path in notebooks.items():
    nb = nbformat.read(path, as_version=4)
    print(f"\n{'='*80}")
    print(f"  {label}: {path}")
    print(f"{'='*80}")
    
    for i, c in enumerate(nb.cells):
        outputs = c.get('outputs', [])
        has_image = any('image/png' in str(o.get('data', {})) for o in outputs)
        
        if has_image or (i < 5 and c.cell_type == 'code'):  # show image cells + first few code cells for context
            marker = "[IMG]" if has_image else "[CODE]"
            # Show full source for understanding dependencies
            print(f"\n--- {marker} Cell {i} ({c.cell_type}) ---")
            # Show just the first 15 lines of source
            lines = c.source.split('\n')
            for line in lines[:20]:
                print(f"  {line}")
            if len(lines) > 20:
                print(f"  ... ({len(lines) - 20} more lines)")
            print(f"  --- outputs: {len(outputs)}, has_image: {has_image} ---")
