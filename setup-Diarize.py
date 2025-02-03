import platform
import shutil
import PyInstaller.__main__

parameters = []

print()

if platform.platform().upper().startswith('WINDOWS'):

    print('Windows')
    
    parameters.append('--add-binary=venv311/Lib/site-packages/asteroid_filterbanks:asteroid_filterbanks')
    parameters.append('--add-binary=venv311/Lib/site-packages/lightning_fabric:lightning_fabric')
    parameters.append('--add-binary=venv311/Lib/site-packages/pyannote:pyannote')
    parameters.append('--add-binary=venv311/Lib/site-packages/speechbrain:speechbrain')
elif platform.platform().upper().startswith('MACOS'):

    print('macOS')
    
    parameters.append('--add-binary=venv311/lib/python3.11/site-packages/asteroid_filterbanks:asteroid_filterbanks')
    parameters.append('--add-binary=venv311/lib/python3.11/site-packages/lightning_fabric:lightning_fabric')
    parameters.append('--add-binary=venv311/lib/python3.11/site-packages/pyannote:pyannote')
    parameters.append('--add-binary=venv311/lib/python3.11/site-packages/speechbrain:speechbrain')

print()

parameters.append('--name=Diarize')
parameters.append('Diarize.py')

PyInstaller.__main__.run(parameters)

## shutil.copyfile('pyannote_access_token.txt', 'dist/Diarize/pyannote_access_token.txt')
shutil.copytree('Media', 'dist/Diarize/Media')
shutil.copytree('models', 'dist/Diarize/models')
