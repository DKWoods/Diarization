# Diarization and Distribution
## A simple example of distributing python software that uses pyannote.audio for diarization
The pyannote.audio module makes diarization (speaker identification) quite easy, and the results appear to be reasonably accurate. 
However, distributing a python program to non-technical users that includes such code proved elusive.  I found many online posts that essentially implied 
that distribution of the pytorch modules using PyInstaller couldn't be done.  In addition, I found it challenging to successfully embed pyannote's libraries, 
eliminating the need to explain to my users how to supply a pyannote access token.
### The usual caveats
This works for me.  Your mileage may vary.  I welcome your input in an effort to expand this example to cover more circumstances.  
* So far, I've been working on Windows 11.  I plan to tackle macOS soon.  I don't know if I'll get to Linux.
* My main Windows computer has a GPU that uses CUDA 12.  If you need CUDA 11, you will need a different version of PyTorch.
* I'm using Python 3.11.6, pyannote.audio 3.3.2, PyTorch 2.6.0, pyannote models "speaker-diarization-3.1" and "segmentation-3.0", and PyInstaller 6.11.1.
These are all current module versions as I write this.
* I've tested the distributable executable on Windows 10 and Windows 11.  One test computer is an 8 year old laptop with a CUDA 12 GPU, and another is a very old
(> 10 years) desktop with an NVidia GeForce GT 545 video card.  No GPU is accessible on this second test computer, so it used "cpu" rather than "cuda" during
diarization.
* Processing time varies widely between computers.
## Using this code sample
At present, create a virtual environment and use pip to import requirements.txt.  Then run "Diarize.py" to see what the program does.  There is a boolean variable 
called EMBED that switches between the normal mode that downloads models from HuggingFace and the embedded or "offline" mode.  You will need to create a file called
"pyannote_access_token.txt" that contains your pyannote access token to use the normal mode.  Then use "setup-Diarize.py" to create a distributable build.  
Alternately, you can use pip to import "requirements with faster whisper.txt" and use "Diarized_Transcript.py".  At this time, this program requires a
pyannote access token for normal on-line diarization and will mix diarization information with transcription information created by faster-whisper.  I have not
written a PyInstaller file for this version yet.  
Unfortunately, two of the four sample audio files this sample user are too large for GitHub.  
## What I've learned:
### Embedding torch libraries
I found several descriptions of how to embed the pyannote models, including 
[pyannote.audio's tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb)'s section for offline use. 
While these descriptions seemed straight forward, I could not get them to work.  They assumed a little too much, said a little too little, and were 
unclear about what exactly needed to be included and edited.  Further, there appears to be just a touch of magic in how one renames the models, 
and I was using model names that made sense to me rather than what is required.  I eventually stumbled upon a bug report that made clear how
important the precise names of the model files were.
* I created a directory called "models".
* I copied the "config.yaml" file from [pyannote's speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) module, which I edited
and renamed "speaker-diarization-3.1-config.yaml".  (This file name is probably not mandatory.  See the file in the repository for details.)
* I copied the "pytorch_model.bin" file from [pyannote's wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
module, which I renamed **pyannote_model_wespeaker-voxceleb-resnet34-LM.bin**.  (I have not included this file in my repository.)
* I copied the "pytorch_model.bin" file from [oyannote's segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) module, which I
renamed **pytorch_model_segmentation-3.0.bin**.  (I have not included this file in my repository.)
* You **must** use these exact file names for the two models.  Otherwise, as I understand it, the models won't load properly otherwise.
* I then changed the pyannote Pipeline command in my code from:
  ```
  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN)
  ```
  to:
  ```
  pipeline = Pipeline.from_pretrained("models/speaker-diarization-3.1-config.yaml")
  ```
### Distribution with PyInstaller
PyInstaller could not initially build an executable for my sample code that would run.  This may not be the optimal way to accomplish this task, but this is 
what worked after several attempts that should have worked in fact did not.  Suggestions for improving this are welcome.
To test the PyInstaller executable on Windows, I loaded a Command Prompt and ran the executable there.  This process allowed me to see where the executable 
was failing. 
This revealed 4 python modules that needed to be included in the distribution but were not.  See setup-Diarize.py.  I added the following to my PyInstaller 
setup command:
```
--add-binary=venv311/Lib/site-packages/asteroid_filterbanks:asteroid_filterbanks
--add-binary=venv311/Lib/site-packages/lightning_fabric:lightning_fabric
--add-binary=venv311/Lib/site-packages/pyannote:pyannote
--add-binary=venv311/Lib/site-packages/speechbrain:speechbrain
```

Following the PyInstaller call, I added the following to copy the necessary model files to the distribution:
```
shutil.copytree('models', 'dist/Diarize/models')
```

I hope this provides enough information that you can figure out how to get this working for you.
