# import Python modules
import os
import pickle
import sys
import time
import traceback

# try:
#     # Faster Whisper is required for loading the transcription data, even though we don't actively transcribe in this module
#     import faster_whisper
#     # We need to make sure the version of Faster Whisper is compatible with our data
#     if faster_whisper.version.__version__ != '0.10.1':
#         raise()
# except:
#     print('This module requires faster_whisper version 0.10.1.')
#     print('Please use "python -m pip install faster_whisper==0.10.1"')
#     sys.exit(1)

def diarize(file_name):
    """ If the diarization file does not already exist, this code will extract the data from the audio file """
    
    # Embedding does not work at this time.
    EMBED = True

    # Import modules required for diarization.  
    try:
        from pyannote.audio import Pipeline
    except:
        
        exc = sys.exc_info()
        print(exc[0])
        print(exc[1])
        traceback.print_exc()
        print()
        
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except:
        
        exc = sys.exc_info()
        print(exc[0])
        print(exc[1])
        traceback.print_exc()
        print()
        
    try:
        import torch
    except:
        
        exc = sys.exc_info()
        print(exc[0])
        print(exc[1])
        traceback.print_exc()
        print()
        
    try:
        import torchaudio
    except:
        
        exc = sys.exc_info()
        print(exc[0])
        print(exc[1])
        traceback.print_exc()
        print()
        
#        # NOTE:  pytorch installs torch and torchaudio on import, so these modules to not require their own error checking.
#        print('This module requires pyannote.audio.  But loading this with GPU support takes two steps.')
#        print('If you have a compatible GPU, go to "https://pytorch.org/" and use the "Install PyTorch" ')
#        print('tool to determine the correct installation command for your computer.')
#        print('Then use "python -m pip install pyannote.audio".')
        sys.exit(1)

    if not EMBED:
        try:
            # Define the pyannote access token.  
            f = open('pyannote_access_token.txt', 'rb')
            ACCESS_TOKEN = f.read().strip()
            f.close()
        except:
        
            exc = sys.exc_info()
            print(exc[0])
            print(exc[1])
            traceback.print_exc()
            print()
        
            print('To process speaker identification, you need a "pyannote/segmentation" access token.')
            print('Go to "https://huggingface.co/pyannote/segmentation" to get a token and place it in ')
            print('a text file called "pyannote access token.txt" in this directory.')
            print()
            sys.exit(1)
    
        print()
        print('HuggingFace Pipeline')
        print()
        
        # Define the pyannote.audio pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=ACCESS_TOKEN)
    else:
        
        print()
        print('EMBEDDED PIPELINE')
        print()
        
        # Starting to explore embedding these libraries, but I haven't cracked it yet.
        pipeline = Pipeline.from_pretrained("models/speaker-diarization-3.1-config.yaml")

    # Determine whether a CUDA (Win, Linux) or MPS (macOS) GPU is available to speed processing
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = torch.device("cpu")

    # Link the pipeline to the GPU device
    pipeline.to(device)
    # Load the waveform data using torchaudio
    waveform, sample_rate = torchaudio.load(file_name)

    print()
    print('Running diarization using "{0}".  This may take a while.'.format(device))
    print()
    
    # Note the start time
    now = time.time()
    with ProgressHook() as hook:
        # Get diarization data from the pyannote.audio pipeline
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    
    print()
    print('Diarization took {0:5.1f} using {1}.'.format(time.time() - now, device))
    print()
    
    # Convert the data to a dictionary in the form we need
    data = {}
    for turn, turnLabel, speaker in diarization.itertracks(yield_label=True):
        # Convert times to ms from start of file
        start = int(turn.start * 1000)
        end = int(turn.end * 1000)
        # Store the speaker ID for the defined time span.  This data is less straight-forward than one might wish.
        data[(start, end)] = speaker
    
    return data

# Define the media directory
file_dir = "Media"

# Get input from the user saying which of 4 pre-defined options they want.  (Options limited for testing.)
dataFileNum = '0'
while not dataFileNum in ['1', '2', '3', '4']:
    print()
    print('Please select from the following options:')
    print(" 1 - Jeanine's Breakfast (2 to 7 seconds)")
    print(" 2 - Heather Stewart - BBC Radio (9 seconds)")
    print(" 3 - Classroom data (66 seconds)")
    print(" 4 - Presidential Debate, October 7, 2008")
    dataFileNum = input()
print()

# Get the appropriate base file name.  (Yes, I could have used a list or dictionary.)
# Files 3 and 4 were selected because they have more speakers than most files
match dataFileNum:
    # Janine's Breakfast
    case '1':
        file_base = "Jeanine's Breakfast"
    # Heather Stewart
    case '2':
        file_base = "gdn.bus.090625.tm.Heather-Stewart2"
    # Volume
    case '3':
        file_base = "Volume"
    # Presidential Debate 20081007
    case '4':
        file_base = "20081007 Presidential Debate-Analysis"

# Determine the needed names for the audio file, the transcript file, and the diarization file
audio_file = os.path.join(file_dir, file_base + ".wav")
transcript_file = os.path.join(file_dir, file_base + "-Transcription.pkl")
diarization_file = os.path.join(file_dir, file_base + "-Diarization.pkl")

# If the diarization file exists ...
if os.path.exists(diarization_file):
    # ... open it and load the diarization data
    df = open(diarization_file, "rb")
    diar_dict_data = df.read()
    df.close()
    diar_dict = pickle.loads(diar_dict_data)
# If the diarization file does not exist ...
else:
    # ... make sure the audio file exists, and exit if not
    if not os.path.exists(audio_file):
        print('Audio file "{0}" not found.'.format(audio_file))
        print()
        sys.exit(1)

    # Get the diarization data from the file
    diar_dict = diarize(audio_file)

    # Open and save the diarization file for future use
    diar_dict_data = pickle.dumps(diar_dict)
    df = open(diarization_file, "wb")
    df.write(diar_dict_data)
    df.close()

# Sort the diarization data's keys
diar_keys = sorted(diar_dict.keys())

# Print the diarization data
print('Diarization:')
for (start, end) in diar_keys:
    print("({0:6}, {1:6}) : '{2}',".format(start, end, diar_dict[(start, end)]))
print()


print('Done')
