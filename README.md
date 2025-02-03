# Diarization and Distribution
## A simple example of distributing python software that uses pyannote.audio for diarization
The pyannote.audio module makes diarization (speaker identification) quite easy, and the results appear to be reasonably accurate. 
However, distributing a python program to non-technical users that includes such code proved elusive.  I found many online posts that essentially said 
that distribution of the pytorch modules couldn't be done.  In addition, I found it challenging to embed pytorch's AI libraries, eliminating the need 
to supply a pytorch access token.
### The usual caveats
This works for me.  Your mileage may vary.  I invite you to help me expand this example to cover more circumstances.  
* My main computer has a GPU that uses CUDA 12.
* For now, I've been working on Windows 11.  I plan to tackle macOS soon.  I don't know if I'll get to Linux.
* I'm using Python 3.11.6, pyannote.audio 3.3.2, PyTorch 2.6.0, models speaker-diarization-3.1 and segmentation-3.0 from pytorch, and PyInstaller 6.11.1.  These are all the current versions as I write this.
* I've tested the distribution code on Windows 10 and Windows 11.  One computer is an 8 year old laptop with a CUDA 12 GPU, and another is a very old
(> 10 years) desktop with an NVidia GeForce GT 545 video card.  No GPU is available on this computer, so it used "cpu" rather than "cuda" during
diarization.
## What I've learned:
### Embedding torch libraries
I found several descriptions of how to embed the AI models, including 
[pyannote.audio's tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb)'s section for offline use. 
While these seemed simple enough, 
I could not get them to work.  They assumed a little too much, and were unclear about what exactly needed to be included and edited.  Further, there
is just a touch of magic in how one renames the models, and I was using model names that made sense to me rather than containing information I hadn't 
figured out was essential.
* I created a directory called "models".
* I copied the "config.yaml" file from [pyannote's speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) module, which I edited
and renamed "speaker-diarization-3.1-config.yaml".  (This file name is probably not mandatory.)
* I copied the "pytorch_model.bin" file from [pyannote's wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
module, which I renamed **pyannote_model_wespeaker-voxceleb-resnet34-LM.bin**.
* I copied the "pytorch_model.bin" file from [oyannote's segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) module, which I
renamed **pytorch_model_segmentation-3.0.bin**.
* You **must** use these exact file names for the two models.  Otherwise, as I understand it, the models won't load properly because they need to know where
they came from and use the file name information to sort that out.
* I then changed my pyannote Pipeline call from:
  > pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=ACCESS_TOKEN)

  to:
  > pipeline = Pipeline.from_pretrained("models/speaker-diarization-3.1-config.yaml")

### Distribution with PyInstaller
PyInstaller could not package my little sample up in a way that would run, at least not without my intervention.  
