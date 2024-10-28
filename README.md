# KIREPRO1PE

Research project ITU

vscode extensions to deal with python:
- ruff for linting (pylinter is a piece of shit extension)


activate venv
```
cd src
source venv/bin/activate
pip install -r requirements.txt
```

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
