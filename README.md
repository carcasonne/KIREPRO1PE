# KIREPRO1PE

Research project ITU

vscode extensions to deal with python:
- ruff for linting (pylinter is a piece of shit extension)

Built and tested using python version 3.12.7

activate venv
```
cd src
source venv/bin/activate
pip install -r requirements.txt
```

# Dataset

`audio_files_samples` and `audio_files_250` contains files from the
Real or Fake dataset from [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?),  specifically the `for-2sec` folder.

`audio_files_samples` contains 40 training, testing and validation files equally 
split into Real and Fake audio. For testing purposes only.

`audio_files_250` contains 250 audio files split into training and testing
in a 4 : 1 split, with equal amount of Real and Fake audio. 
(Also has a validation folder to make the program work.)


# graphic cards requirements

log from running with batch size 16:
```
GPU Memory at start of batch:
Allocated: 2556.92 MB
Cached: 6172.00 MB
Images shape: torch.Size([8, 3, 462, 775])
Labels shape: torch.Size([8])
Memory per batch: 32.78 MB
```

so you need a lot of vram
