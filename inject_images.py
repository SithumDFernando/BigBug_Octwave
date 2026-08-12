"""
inject_images.py — Inject full code+output cells from source notebooks into the submission notebook.

This script copies complete cells (code + pre-rendered outputs) from the EDA, Preprocessing, 
and Evaluation notebooks into TeamBigBug_OctWave_Submission.ipynb at the appropriate narrative
positions. No re-execution is needed — all images are already baked into the .ipynb JSON as base64.

Usage: python inject_images.py
"""
import nbformat
import copy
import re
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBMISSION_NB = 'TeamBigBug_OctWave_Submission.ipynb'
EDA_NB        = 'notebooks/01_EDA.ipynb'
PREPROCESS_NB = 'notebooks/02_Data_Preprocessing.ipynb'
EVAL_NB       = 'notebooks/04_Evaluation_and_Prediction.ipynb'

# Path fixups: source notebooks live in notebooks/, submission lives at root
PATH_FIXUPS = [
    ("'../data/raw/",        "'data/raw/"),
    ("'../data/processed/",  "'data/processed/"),
    ("'../models/",          "'models/"),
    ("'../outputs/",         "'outputs/"),
    ('"../data/raw/',        '"data/raw/'),
    ('"../data/processed/',  '"data/processed/'),
    ('"../models/',          '"models/'),
    ('"../outputs/',         '"outputs/'),
]


def fix_paths(cell):
    """Fix relative paths in a cell's source to work from project root."""
    cell = copy.deepcopy(cell)
    source = cell['source']
    # Handle both string and list source formats
    if isinstance(source, list):
        source = ''.join(source)
    for old, new in PATH_FIXUPS:
        source = source.replace(old, new)
    cell['source'] = source
    return cell


def get_cell(nb, idx):
    """Get a deep copy of a cell from a notebook, with paths fixed."""
    return fix_paths(nb.cells[idx])


def split_markdown_at(cell, header_text):
    """
    Split a markdown cell's source at a line starting with header_text.
    Returns (before_cell, after_cell) — both are new markdown cells.
    If header_text is not found, returns (cell, None).
    """
    source = cell['source']
    # Source can be a string or list of strings
    if isinstance(source, list):
        source = ''.join(source)
    
    # Split into lines (preserving newlines)
    source_lines = source.split('\n')
    
    # Find the line that starts with header_text
    split_idx = None
    for i, line in enumerate(source_lines):
        if line.strip().startswith(header_text):
            split_idx = i
            break
    
    if split_idx is None:
        return cell, None
    
    before_text = '\n'.join(source_lines[:split_idx]).rstrip()
    after_text = '\n'.join(source_lines[split_idx:])
    
    before_cell = nbformat.v4.new_markdown_cell(before_text)
    after_cell = nbformat.v4.new_markdown_cell(after_text)
    
    return before_cell, after_cell


def renumber_execution_counts(nb):
    """Renumber all code cell execution_counts sequentially."""
    count = 1
    for cell in nb.cells:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = count
            # Also renumber execute_result outputs
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'execute_result':
                    output['execution_count'] = count
            count += 1


def main():
    print("Loading notebooks...")
    sub_nb  = nbformat.read(SUBMISSION_NB, as_version=4)
    eda_nb  = nbformat.read(EDA_NB, as_version=4)
    pre_nb  = nbformat.read(PREPROCESS_NB, as_version=4)
    eval_nb = nbformat.read(EVAL_NB, as_version=4)
    
    print(f"  Submission: {len(sub_nb.cells)} cells")
    print(f"  EDA:        {len(eda_nb.cells)} cells")
    print(f"  Preprocess: {len(pre_nb.cells)} cells")
    print(f"  Evaluation: {len(eval_nb.cells)} cells")
    
    # -----------------------------------------------------------------------
    # The current submission notebook has this structure (0-indexed):
    #   0: Markdown — Sections 1-5 (Context, EDA, Cleaning, Feature Eng)
    #   1: Code    — Feature engineering pipeline
    #   2: Markdown — Section 6 (Model Training)
    #   3: Code    — Model training (12 models)
    #   4: Markdown — Section 7 (Ensemble)
    #   5: Code    — Ensemble construction
    #   6: Markdown — Section 8 (Advanced Ensemble / Stacking)
    #   7: Code    — Advanced ensemble
    #   8: Markdown — Sections 9-11 (Final Selection, Evaluation, Results)
    # -----------------------------------------------------------------------
    
    new_cells = []
    
    # === PART 1: Split the big first markdown cell and interleave EDA plots ===
    big_md = sub_nb.cells[0]  # The huge markdown cell (Sections 1-5)
    
    # Split at "## 3. Phase 2: Exploratory Data Analysis (EDA)"
    before_eda, rest = split_markdown_at(big_md, "## 3. Phase 2: Exploratory Data Analysis (EDA)")
    if rest is None:
        print("ERROR: Could not find '## 3. Phase 2: Exploratory Data Analysis (EDA)' in first markdown cell!")
        return
    
    # Add Sections 1-2 markdown
    new_cells.append(before_eda)
    
    # Add EDA imports + data loading (cells 2 and 4 from EDA notebook)
    # Cell 2 = imports, Cell 4 = data loading
    new_cells.append(get_cell(eda_nb, 2))  # imports + style setup
    new_cells.append(get_cell(eda_nb, 4))  # data loading (train_df, test_df)
    
    # Split rest at "## 4. Phase 3: Data Cleaning"
    # But first, we need to inject plots between subsections of Phase 2
    
    # Split at "### 3.2 Outlier Analysis (IQR Method)"
    sec3_intro, rest = split_markdown_at(rest, "### 3.2 Outlier Analysis (IQR Method)")
    if rest is None:
        print("ERROR: Could not find '### 3.2 Outlier Analysis (IQR Method)'!")
        return
    new_cells.append(sec3_intro)  # Sections 3.0-3.1
    
    # Split at "### 3.3 Zero/Edge Value Analysis"
    sec3_2, rest = split_markdown_at(rest, "### 3.3 Zero/Edge Value Analysis")
    if rest is None:
        print("ERROR: Could not find '### 3.3 Zero/Edge Value Analysis'!")
        return
    new_cells.append(sec3_2)  # Section 3.2 (Outlier Analysis text)
    
    # NOTE: No outlier plot cell — the outlier analysis is text/table-based (IQR counts)
    
    # Split at "### 3.4 Fraud Signal Discovery"
    sec3_3, rest = split_markdown_at(rest, "### 3.4 Fraud Signal")
    if rest is None:
        print("ERROR: Could not find '### 3.4 Fraud Signal Discovery'!")
        return
    new_cells.append(sec3_3)  # Section 3.3
    
    # Split at "### 3.5 Correlation Analysis"
    sec3_4, rest = split_markdown_at(rest, "### 3.5 Correlation Analysis")
    if rest is None:
        print("ERROR: Could not find '### 3.5 Correlation Analysis'!")
        return
    new_cells.append(sec3_4)  # Section 3.4 (Fraud Signal Discovery text)
    
    # Insert fraud signal discovery plots
    new_cells.append(get_cell(eda_nb, 22))  # Side-by-side boxplots
    new_cells.append(get_cell(eda_nb, 24))  # Binary feature fraud rates
    new_cells.append(get_cell(eda_nb, 26))  # Merchant category distribution
    
    # Split at "### 3.6 Train vs Test"
    sec3_5, rest = split_markdown_at(rest, "### 3.6 Train vs Test")
    if rest is None:
        print("ERROR: Could not find '### 3.6 Train vs Test'!")
        return
    new_cells.append(sec3_5)  # Section 3.5 (Correlation Analysis text)
    
    # Insert correlation plots
    new_cells.append(get_cell(eda_nb, 28))  # Correlation heatmap
    new_cells.append(get_cell(eda_nb, 30))  # Target correlation bar chart
    
    # Insert deep dive plots
    new_cells.append(get_cell(eda_nb, 32))  # Transaction hour
    new_cells.append(get_cell(eda_nb, 34))  # Device trust score
    new_cells.append(get_cell(eda_nb, 36))  # Amount
    new_cells.append(get_cell(eda_nb, 38))  # Velocity
    new_cells.append(get_cell(eda_nb, 40))  # Foreign x Location interaction
    new_cells.append(get_cell(eda_nb, 42))  # Pairplot
    
    # Split at "### 3.7 Critical Visualizations Performed"
    sec3_6, sec3_7_and_rest = split_markdown_at(rest, "### 3.7 Critical Visualizations")
    if sec3_7_and_rest is None:
        print("ERROR: Could not find '### 3.7 Critical Visualizations'!")
        return
    new_cells.append(sec3_6)  # Section 3.6 (Train vs Test text)
    
    # Insert Train vs Test comparison plots (these were incorrectly placed before)
    new_cells.append(get_cell(eda_nb, 20))  # Train+Test histogram overlay (density-normalized)
    new_cells.append(get_cell(eda_nb, 46))  # Train vs Test stats bar charts (mean/median/std)
    
    # Reword Section 3.7 — it used to say "must be reproduced", now they ARE reproduced above
    sec3_7_src = sec3_7_and_rest['source'] if isinstance(sec3_7_and_rest['source'], str) else ''.join(sec3_7_and_rest['source'])
    sec3_7_src = sec3_7_src.replace(
        "### 3.7 Critical Visualizations Performed\n"
        "The following visualizations were instrumental in discovering the patterns above and must be reproduced in the final notebook:",
        "### 3.7 Summary of Visualizations Performed\n"
        "The following visualizations were performed as part of our EDA (shown above) and were instrumental in discovering the patterns that guided our feature engineering and modeling decisions:"
    )
    sec3_7_and_rest = nbformat.v4.new_markdown_cell(sec3_7_src)
    
    # Split at "## 4. Phase 3: Data Cleaning & Validation"
    sec3_7_final, rest = split_markdown_at(sec3_7_and_rest, "## 4. Phase 3: Data Cleaning")
    if rest is None:
        print("ERROR: Could not find '## 4. Phase 3: Data Cleaning'!")
        return
    new_cells.append(sec3_7_final)  # Section 3.7 (reworded)
    
    # Split at "## 5. Phase 4: Feature Engineering"  
    sec4, rest = split_markdown_at(rest, "## 5. Phase 4: Feature Engineering")
    if rest is None:
        print("ERROR: Could not find '## 5. Phase 4: Feature Engineering'!")
        return
    new_cells.append(sec4)  # Section 4 (Data Cleaning)
    
    # Now add the rest of section 5 (Feature Engineering) 
    # This is the remaining text before the existing code cell
    new_cells.append(rest)  # Section 5 (Feature Engineering text)
    
    # === PART 2: Existing feature engineering code cell ===
    new_cells.append(sub_nb.cells[1])  # Existing code cell with outputs
    
    # Insert post-engineering correlation with target (Preprocessing cell 40)
    new_cells.append(get_cell(pre_nb, 40))
    
    # === PART 3: Model Training section (unchanged) ===
    new_cells.append(sub_nb.cells[2])  # Section 6 markdown
    new_cells.append(sub_nb.cells[3])  # Model training code
    
    # === PART 4: Ensemble section (unchanged) ===
    new_cells.append(sub_nb.cells[4])  # Section 7 markdown
    new_cells.append(sub_nb.cells[5])  # Ensemble code
    
    # === PART 5: Advanced Ensemble section (unchanged) ===
    new_cells.append(sub_nb.cells[6])  # Section 8 markdown
    new_cells.append(sub_nb.cells[7])  # Advanced ensemble code
    
    # === PART 6: Split final markdown and inject evaluation plots ===
    final_md = sub_nb.cells[8]  # Sections 9-11
    
    # Split at "## 10. Final Evaluation, Visualization & Submission Strategy"
    sec9, rest = split_markdown_at(final_md, "## 10. Final Evaluation")
    if rest is None:
        print("ERROR: Could not find '## 10. Final Evaluation'!")
        return
    new_cells.append(sec9)  # Section 9 (Final Model Selection)
    
    # Split at "### 10.2 Dual Submission"
    sec10_intro, rest = split_markdown_at(rest, "### 10.2 Dual Submission")
    if rest is None:
        print("ERROR: Could not find '### 10.2 Dual Submission'!")
        return
    new_cells.append(sec10_intro)  # Section 10 intro + 10.1
    
    # Inject evaluation setup cells (these have text outputs too — leaderboard, etc.)
    new_cells.append(get_cell(eval_nb, 2))   # Imports + plot setup
    new_cells.append(get_cell(eval_nb, 3))   # Load data  
    new_cells.append(get_cell(eval_nb, 4))   # Load model results log + leaderboard table
    new_cells.append(get_cell(eval_nb, 6))   # Load top models
    new_cells.append(get_cell(eval_nb, 7))   # Load ensemble metadata
    new_cells.append(get_cell(eval_nb, 9))   # OOF predictions
    new_cells.append(get_cell(eval_nb, 10))  # Classification reports
    
    # Inject evaluation plot cells
    new_cells.append(get_cell(eval_nb, 11))  # Confusion matrices
    new_cells.append(get_cell(eval_nb, 12))  # ROC curves
    new_cells.append(get_cell(eval_nb, 13))  # Precision-Recall curves
    
    # Threshold analysis needs helper function (cell 15) then plot (cell 16)
    new_cells.append(get_cell(eval_nb, 15))  # threshold_analysis() function
    new_cells.append(get_cell(eval_nb, 16))  # Threshold sweep plot
    
    # Feature importance needs setup (cell 20) then plot (cell 21)
    new_cells.append(get_cell(eval_nb, 20))  # Feature importance extraction
    new_cells.append(get_cell(eval_nb, 21))  # Feature importance bar charts
    
    # Add remaining markdown (10.2-11)
    new_cells.append(rest)
    
    # === Finalize ===
    sub_nb.cells = new_cells
    
    # Renumber execution counts
    renumber_execution_counts(sub_nb)
    
    # Save
    nbformat.write(sub_nb, SUBMISSION_NB)
    
    print(f"\nDone! Modified notebook saved to {SUBMISSION_NB}")
    print(f"  Total cells: {len(sub_nb.cells)}")
    code_cells = sum(1 for c in sub_nb.cells if c['cell_type'] == 'code')
    md_cells = sum(1 for c in sub_nb.cells if c['cell_type'] == 'markdown')
    img_cells = sum(1 for c in sub_nb.cells if c['cell_type'] == 'code' and 
                    any('image/png' in str(o.get('data', {})) for o in c.get('outputs', [])))
    print(f"  Code cells: {code_cells}")
    print(f"  Markdown cells: {md_cells}")
    print(f"  Cells with images: {img_cells}")


if __name__ == '__main__':
    main()
