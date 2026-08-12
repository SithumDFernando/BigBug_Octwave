import nbformat

nb = nbformat.read('TeamBigBug_OctWave_Submission.ipynb', as_version=4)
print(f"Valid notebook: {len(nb.cells)} cells")

for i, c in enumerate(nb.cells):
    ct = c['cell_type']
    ec = c.get('execution_count', '-')
    outputs = c.get('outputs', [])
    has_img = any('image/png' in str(o.get('data', {})) for o in outputs)
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    preview = src[:80].replace('\n', ' | ')
    # Handle unicode for Windows console
    preview = preview.encode('ascii', 'replace').decode('ascii')
    
    img_marker = "IMG" if has_img else "   "
    print(f"  Cell {i:2d}: {ct:8s} exec={str(ec):>3s} [{img_marker}] {preview}")

# Summary
code_cells = sum(1 for c in nb.cells if c['cell_type'] == 'code')
md_cells = sum(1 for c in nb.cells if c['cell_type'] == 'markdown')
img_cells = sum(1 for c in nb.cells if c['cell_type'] == 'code' and 
                any('image/png' in str(o.get('data', {})) for o in c.get('outputs', [])))
print(f"\nSummary: {code_cells} code, {md_cells} markdown, {img_cells} with images")
