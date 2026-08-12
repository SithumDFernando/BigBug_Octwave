import nbformat

nb = nbformat.read('notebooks/04_Evaluation_and_Prediction.ipynb', as_version=4)
# Show cells 14-15 and 18-20 (threshold_analysis helper and feature importance setup)
for i in [14, 15, 18, 19, 20]:
    if i < len(nb.cells):
        c = nb.cells[i]
        outputs = c.get('outputs', [])
        has_image = any('image/png' in str(o.get('data', {})) for o in outputs)
        print(f"\n--- Cell {i} ({c.cell_type}) has_image={has_image} ---")
        print(c.source[:500])
        print(f"--- end (total {len(c.source)} chars) ---")
