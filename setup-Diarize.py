import shutil
import PyInstaller.__main__

parameters = []
parameters.append('--add-binary=venv311/Lib/site-packages/asteroid_filterbanks:asteroid_filterbanks')
parameters.append('--add-binary=venv311/Lib/site-packages/lightning_fabric:lightning_fabric')
parameters.append('--add-binary=venv311/Lib/site-packages/pyannote:pyannote')
parameters.append('--add-binary=venv311/Lib/site-packages/speechbrain:speechbrain')
parameters.append('--name=Diarize')
parameters.append('Diarize.py')

PyInstaller.__main__.run(parameters)

## shutil.copyfile('pyannote_access_token.txt', 'dist/Diarize/pyannote_access_token.txt')
shutil.copytree('Media', 'dist/Diarize/Media')
shutil.copytree('models', 'dist/Diarize/models')