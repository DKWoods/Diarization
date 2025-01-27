# import Python modules
import os
import pickle
import time

try:
    # Faster Whisper is required for loading the transcription data, even though we don't actively transcribe in this module
    import faster_whisper
    # We need to make sure the version of Faster Whisper is compatible with our data
    if faster_whisper.version.__version__ != '0.10.1':
        raise()
except:
    print('This module requires faster_whisper version 0.10.1.')
    print('Please use "python -m pip install faster_whisper==0.10.1"')
    exit(1)

def diarize(file_name):
    """ If the diarization file does not already exist, this code will extract the data from the audio file """

    # Import modules required for diarization.  
    try:
        from pyannote.audio import Pipeline
        import torch
        import torchaudio
    except:
        # NOTE:  pytorch installs torch and torchaudio on import, so these modules to not require their own error checking.
        print('This module requires pyannote.audio.  But loading this with GPU support takes two steps.')
        print('If you have a compatible GPU, go to "https://pytorch.org/" and use the "Install PyTorch" ')
        print('tool to determine the correct installation command for your computer.')
        print('Then use "python -m pip install pyannote.audio".')
        exit(1)

    try:
        # Define the pyannote access token.  
        f = open('pyannote access token.txt', 'rb')
        ACCESS_TOKEN = f.read().strip()
        f.close()
    except:
        print('To process speaker identification, you need a "pyannote/segmentation" access token.')
        print('Go to "https://huggingface.co/pyannote/segmentation" to get a token and place it in ')
        print('a text file called "pyannote access token.txt" in this directory.')
        print()
        exit(1)
    
    # Define the pyannote.audio pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",  # WAS @2.1
                                    use_auth_token=ACCESS_TOKEN)
    # Starting to explore embedding these libraries, but I haven't cracked it yet.
#    pipeline = Pipeline.from_pretrained("config_diarize.yaml")

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
    print("Running diarization.  This may take a while.")
    print()
    
    # Note the start time
    now = time.time()
    # Get diarization data from the pyannote.audio pipeline
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})  # , min_speakers=0, max_speakers=100)
    
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
    print(" 1 - Jeanine's Breakfast")
    print(" 2 - Heather Stewart - BBC Radio")
    print(" 3 - Presidential Debate, October 7, 2008")
    print(" 4 - Classroom data")
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
    # Presidential Debate 20081007
    case '3':
        file_base = "20081007 Presidential Debate-Analysis"
    # Volume
    case '4':
        file_base = "Volume"

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
        exit(1)

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

# If the transcript file exists ...
if os.path.exists(transcript_file):
    # ... open it and load the transcription data
    f = open(transcript_file, 'rb')
    fd = f.read()
    f.close()
    # This step requires the faster_whisper module
    trans = pickle.loads(fd)

    # The transcription file provides words, but we need sentences for diarization comparisons
    # Initialize a dictionary of sentences    
    sentences = []
    # Initialize the start time of the individual sentence
    sentenceStart = -1
    # Initialize a string for an individual sentence
    sentence = ''
    
    # For each line in the transcript data ...
    for line in trans:
        # For each word object in the line ...
        for word_rec in line:
            # If we need to note the start time, for a sentence ...
            if sentenceStart == -1:
                # ... convert the word start time to milliseconds
                sentenceStart = int(word_rec.start * 1000)
            # Add the word to the sentence string
            sentence += word_rec.word + ' '
            # If the word ends with punctuation ...
            if word_rec.word.rstrip()[-1] in ('.', '?', '!'):
                # ... add the sentence to the sentences dictionary, keyed to sentence start and last word end
                sentences.append((sentenceStart, int(word_rec.end * 1000), sentence))
                # Signal we need to capture the next time code
                sentenceStart = -1
                # Reset the individual sentence string
                sentence = ''
    
    # Print the transcription sentence content 
    print('Transcription:')
    for (start, end, sentence) in sentences:
        print('{0}\t{1}\t{2}'.format(start, end, sentence))
    print()
    
    # Unfortunately, the start and end points from diarization (speaker identifiers) does not match the start and 
    # end times of transcribed sentences.  Not even close in most instances.  
    #
    # The best method I've found so far (I think, subject to more comparisons to the original audio) is to look for
    # the greatest amount of overlap
    
    # Create a variable for showing or hiding the data for comparing diarization sentences and transcription sentences.
    showDiarizationLogic = True
    
    # Now let's build and print the final diarized transcript                
    print('+------------------------------------------------------------------------------------+')
    print('Final Diarized Transcript')
    print()
    
    # For each sentence based on Transcription ...
    for (se_start, se_end, sentence) in sentences:
        if showDiarizationLogic:
            print('  * {0}\t{1}\t{2}'.format(se_start, se_end, sentence))
        # Initialize variables for maximum time overlap and speaker id dictionary on a sentence by sentence basis
        maxVal = 0
        spIds = {}
        # For each Diarization record...  (This is inefficient, as it goes through the whole diarization list every time.)
        for (sp_start, sp_end) in diar_keys:
            # ... note the speaker
            speaker = diar_dict[(sp_start, sp_end)]
            # The Transcription engine (Faster Whisper) marks sentence starts earlier than the Diarization engine (pyannote.audio).
            # Therefore, we can look for overlaps by looking for sentence starts before speaker ID starts and sentence ends
            # later than speaker starts.
            if (se_start <= sp_end) and (se_end >= sp_start):
                # The amount of overlap between the two systems is the smaller end point minus the larger start point 
                overlapAmt = min(se_end, sp_end) - max(se_start, sp_start)
                # If this is the largest overlap identified so far ...
                if overlapAmt > maxVal:
                    # ... note the size of the overlap and the speaker ID 
                    maxVal = overlapAmt
                spIds[overlapAmt] = (speaker, sp_start, sp_end)
                if showDiarizationLogic:
                    print("  * {0}\t{1}\t{2} - '{3}' ({4})".format(sp_start, sp_end, speaker, sentence, overlapAmt))
        if showDiarizationLogic:
            print()
        # If no overlapping segment was found ...
        if maxVal == 0:
            # ... label the speaker as "Unknown"
            spIds[maxVal] = ('Unknown', se_start, se_end)

        print("{0}\t{1}\t{2} - '{3}'".format(se_start, se_end, spIds[maxVal][0], sentence))
        if showDiarizationLogic:
            print()
    print()

# If there's no Transcription file ...
else:
    # ... just report it.  Creation of the Transcription files is beyond the scope of this sample file.
    print('Transcription file not found.')

print('Done')
