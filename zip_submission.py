import zipfile
import os
import shutil

source_nb_path = 'notebooks/Working_Submission.ipynb'
final_nb_path = 'TeamBigBug_OctWave_Submission.ipynb'

if os.path.exists(source_nb_path):
    shutil.copy(source_nb_path, final_nb_path)
    print(f'Successfully copied {source_nb_path} to {final_nb_path}.')
else:
    print(f'Error: Could not find {source_nb_path}.')

zip_path = 'TeamBigBug_OctWave_Submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('TeamBigBug_Report.md', arcname='TeamBigBug_Report.md')
    zipf.write(final_nb_path, arcname=final_nb_path)

print(f'Created zip file {zip_path}')
