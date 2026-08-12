"""Deep dive into specific audit findings."""
import nbformat

eda_nb = nbformat.read('notebooks/01_EDA.ipynb', as_version=4)

# Check EDA cell 14 — the one flagged as train vs test comparison with image
print("=" * 80)
print("EDA Cell 14 (train vs test comparison?) — FULL SOURCE:")
print("=" * 80)
src = eda_nb.cells[14]['source'] if isinstance(eda_nb.cells[14]['source'], str) else ''.join(eda_nb.cells[14]['source'])
print(src[:500])
print("...")

# Also check cells around it for context
print("\n" + "=" * 80)
print("EDA Cell 13 (markdown before cell 14):")
print("=" * 80)
if eda_nb.cells[13]['cell_type'] == 'markdown':
    src = eda_nb.cells[13]['source'] if isinstance(eda_nb.cells[13]['source'], str) else ''.join(eda_nb.cells[13]['source'])
    print(src[:300])

# Check for a dedicated train vs test overlay cell
print("\n" + "=" * 80)
print("SEARCHING EDA for 'train vs test' or 'overlay' or 'test_df' plots:")
print("=" * 80)
for i, c in enumerate(eda_nb.cells):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    src_lower = src.lower()
    has_img = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
    
    # Look for cells that explicitly compare train and test distributions
    if c['cell_type'] == 'code' and 'test_df' in src and has_img:
        print(f"\n--- EDA Cell {i} (uses test_df + has image) ---")
        print(src[:400])
        print("---")
    elif c['cell_type'] == 'markdown' and ('train vs test' in src_lower or 'train and test' in src_lower or 'overlay' in src_lower):
        print(f"\n--- EDA Cell {i} (markdown about train/test) ---")
        print(src[:300])

# Check Cell 44 (threshold_analysis function - no output)
print("\n" + "=" * 80)
print("SUBMISSION Cell 44 — threshold_analysis function (no output):")
print("=" * 80)
sub_nb = nbformat.read('TeamBigBug_OctWave_Submission.ipynb', as_version=4)
src = sub_nb.cells[44]['source'] if isinstance(sub_nb.cells[44]['source'], str) else ''.join(sub_nb.cells[44]['source'])
print(src[:500])
print(f"\nOutputs: {sub_nb.cells[44].get('outputs', [])}")

# Check original eval notebook cell 15 for outputs
eval_nb = nbformat.read('notebooks/04_Evaluation_and_Prediction.ipynb', as_version=4)
print(f"\nOriginal eval cell 15 outputs: {eval_nb.cells[15].get('outputs', [])}")
