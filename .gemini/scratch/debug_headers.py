import nbformat

nb = nbformat.read('TeamBigBug_OctWave_Submission_backup.ipynb', as_version=4)
cell0 = nb.cells[0]
src = cell0['source']

# Check if source is string or list
print(f"Type of source: {type(src)}")
print(f"Length: {len(src)}")

# Find all lines with ## headers
if isinstance(src, str):
    lines = src.split('\n')
else:
    lines = src

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('##'):
        print(f"  Line {i}: {repr(stripped[:80])}")
