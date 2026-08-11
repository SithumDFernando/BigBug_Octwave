import zipfile
import os
import shutil

try:
    import nbformat
    nbs = [
        'notebooks/01_EDA.ipynb',
        'notebooks/02_Data_Preprocessing.ipynb',
        'notebooks/03_Model_Training.ipynb',
        'notebooks/04_Evaluation_and_Prediction.ipynb'
    ]
    merged = nbformat.v4.new_notebook()
    for f in nbs:
        nb = nbformat.read(f, as_version=4)
        merged.cells.extend(nb.cells)
    final_nb_path = 'TeamBigBug_OctWave_Submission.ipynb'
    nbformat.write(merged, final_nb_path)
    print('Merged notebooks successfully.')
except Exception as e:
    print('Could not merge notebooks, falling back to 04_Evaluation_and_Prediction.ipynb. Error:', e)
    final_nb_path = 'TeamBigBug_OctWave_Submission.ipynb'
    shutil.copy('notebooks/04_Evaluation_and_Prediction.ipynb', final_nb_path)

zip_path = 'TeamBigBug_OctWave_Submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('TeamBigBug_Report.md', arcname='TeamBigBug_Report.md')
    zipf.write(final_nb_path, arcname=final_nb_path)

print(f'Created zip file {zip_path}')
