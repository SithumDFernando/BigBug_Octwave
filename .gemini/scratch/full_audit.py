"""Full audit of the submission notebook — check narrative flow, missing elements, issues."""
import nbformat

nb = nbformat.read('TeamBigBug_OctWave_Submission.ipynb', as_version=4)

print("=" * 80)
print("FULL AUDIT: TeamBigBug_OctWave_Submission.ipynb")
print("=" * 80)

issues = []
warnings = []

# --- 1. Check overall structure ---
print("\n--- 1. CELL STRUCTURE ---")
for i, c in enumerate(nb.cells):
    ct = c['cell_type']
    ec = c.get('execution_count', '-')
    outputs = c.get('outputs', [])
    has_img = any('image/png' in str(o.get('data', {})) for o in outputs)
    has_text_out = any(o.get('output_type') == 'stream' for o in outputs)
    has_html_out = any('text/html' in str(o.get('data', {})) for o in outputs)
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    
    # Get first meaningful line for preview
    lines = src.strip().split('\n')
    first_line = lines[0][:100].encode('ascii', 'replace').decode('ascii') if lines else "(empty)"
    
    markers = []
    if has_img: markers.append("IMG")
    if has_text_out: markers.append("TXT")
    if has_html_out: markers.append("HTML")
    marker_str = ','.join(markers) if markers else "---"
    
    print(f"  [{i:2d}] {ct:8s} exec={str(ec):>3s} [{marker_str:>8s}] {first_line}")
    
    # Check for issues
    if ct == 'code' and not outputs:
        warnings.append(f"Cell {i}: Code cell has NO outputs (looks unexecuted)")
    if ct == 'code' and ec is None:
        issues.append(f"Cell {i}: Code cell has no execution_count")

# --- 2. Check narrative flow ---
print("\n--- 2. NARRATIVE FLOW CHECK ---")
# Extract all markdown headers
all_headers = []
for i, c in enumerate(nb.cells):
    if c['cell_type'] == 'markdown':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        for line in src.split('\n'):
            stripped = line.strip()
            if stripped.startswith('## ') or stripped.startswith('### '):
                h = stripped.encode('ascii', 'replace').decode('ascii')
                all_headers.append((i, h))

print("  Headers found:")
for cell_idx, h in all_headers:
    print(f"    Cell {cell_idx:2d}: {h}")

# --- 3. Check for train vs test comparison ---
print("\n--- 3. TRAIN vs TEST COMPARISON CHECK ---")
found_train_test = False
for i, c in enumerate(nb.cells):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    src_lower = src.lower()
    if ('train' in src_lower and 'test' in src_lower and 
        ('distribution' in src_lower or 'comparison' in src_lower or 'overlay' in src_lower or 'drift' in src_lower)):
        has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
        print(f"  Cell {i} ({c['cell_type']}): mentions train/test comparison, has_image={has_img}")
        found_train_test = True

if not found_train_test:
    issues.append("No train vs test distribution comparison PLOT found in notebook!")
    print("  WARNING: No train vs test comparison plot found!")

# --- 4. Check what the EDA narrative promises vs what's delivered ---
print("\n--- 4. PROMISED vs DELIVERED VISUALIZATIONS ---")
# Section 3.7 lists what should be in the notebook
promised = [
    "Numerical feature histograms",
    "Side-by-side boxplots (Fraud vs Legitimate)",
    "Merchant category distribution",
    "Correlation heatmap",
    "Deep dive plots (transaction_hour, device_trust_score, amount, velocity_last_24h)",
    "Feature interaction plot (foreign_transaction x location_mismatch)",
    "Pairplot colored by fraud class",
    "Train vs Test distribution overlay plots",
]

# Count image cells
img_cells = []
for i, c in enumerate(nb.cells):
    if c['cell_type'] == 'code':
        has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
        if has_img:
            src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            first_comment = ""
            for line in src.split('\n'):
                if line.strip().startswith('#'):
                    first_comment = line.strip()[:80].encode('ascii', 'replace').decode('ascii')
                    break
            img_cells.append((i, first_comment))

print(f"  Promised visualizations: {len(promised)}")
for p in promised:
    print(f"    - {p}")
print(f"\n  Image cells found: {len(img_cells)}")
for idx, comment in img_cells:
    print(f"    Cell {idx:2d}: {comment}")

# --- 5. Check for data loading issues ---
print("\n--- 5. DATA DEPENDENCY CHECK ---")
# EDA cells need train_df, test_df - check if they're loaded before first use
first_train_df_use = None
first_train_df_load = None
for i, c in enumerate(nb.cells):
    if c['cell_type'] == 'code':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        if 'train_df' in src:
            if first_train_df_use is None:
                first_train_df_use = i
            if 'read_csv' in src and 'train_df' in src:
                if first_train_df_load is None:
                    first_train_df_load = i

if first_train_df_load is not None and first_train_df_use is not None:
    if first_train_df_load <= first_train_df_use:
        print(f"  OK: train_df loaded in cell {first_train_df_load}, first used in cell {first_train_df_use}")
    else:
        issues.append(f"train_df used in cell {first_train_df_use} BEFORE loaded in cell {first_train_df_load}")

# Check evaluation cells need X_train, loaded_models, etc.
eval_vars = ['X_train', 'y_train', 'loaded_models', 'oof_probas', 'oof_preds', 'eval_models', 
             'top5_names', 'leaderboard', 'ensemble_meta']
for var in eval_vars:
    first_def = None
    first_use = None
    for i, c in enumerate(nb.cells):
        if c['cell_type'] == 'code':
            src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            if var in src:
                if first_use is None:
                    first_use = i
                # Check if this cell defines it (rough heuristic: assignment)
                if f'{var} =' in src or f'{var}[' in src:
                    if first_def is None:
                        first_def = i
    if first_def is not None and first_use is not None:
        if first_def > first_use:
            issues.append(f"Variable '{var}' used in cell {first_use} before defined in cell {first_def}")

# --- 6. Check for the preprocessing cell variable issue ---
print("\n--- 6. PREPROCESSING CORRELATION PLOT CHECK ---")
# Cell 25 (post-engineering correlation) uses train_engineered — is it defined?
for i, c in enumerate(nb.cells):
    if c['cell_type'] == 'code':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        if 'train_engineered' in src:
            has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
            load_in_cell = 'read_csv' in src or 'train_engineered =' in src
            print(f"  Cell {i}: uses train_engineered, defines_it={load_in_cell}, has_image={has_img}")

# --- 7. Check the EDA notebooks for train vs test comparison plot ---
print("\n--- 7. EDA NOTEBOOK: TRAIN vs TEST COMPARISON PLOTS ---")
eda_nb = nbformat.read('notebooks/01_EDA.ipynb', as_version=4)
for i, c in enumerate(eda_nb.cells):
    if c['cell_type'] == 'code':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        src_lower = src.lower()
        has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
        if ('test_df' in src and 'train_df' in src and has_img) or 'overlay' in src_lower or 'train vs test' in src_lower:
            first_line = src.split('\n')[0][:80].encode('ascii', 'replace').decode('ascii')
            print(f"  EDA Cell {i}: has_image={has_img} | {first_line}")

# Also check preprocessing notebook
pre_nb = nbformat.read('notebooks/02_Data_Preprocessing.ipynb', as_version=4)
for i, c in enumerate(pre_nb.cells):
    if c['cell_type'] == 'code':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        src_lower = src.lower()
        has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
        if 'test' in src_lower and 'train' in src_lower and ('overlay' in src_lower or 'comparison' in src_lower or 'drift' in src_lower or 'distribution' in src_lower):
            first_line = src.split('\n')[0][:80].encode('ascii', 'replace').decode('ascii')
            print(f"  Preprocess Cell {i}: has_image={has_img} | {first_line}")

# --- SUMMARY ---
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
if issues:
    print(f"\nISSUES ({len(issues)}):")
    for issue in issues:
        print(f"  [!] {issue}")
else:
    print("\n  No critical issues found.")

if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  [?] {w}")
else:
    print("  No warnings.")
