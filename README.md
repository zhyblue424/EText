# Win-Win: Enhancing Zero-Shot Robustness in VLMs Without Compromising Generalization

## CLIP Model

### Environment setup:

install virtual environment:
`pip install virtualenv`

`virtualenv EText`

`source EText/venv/bin/activate`

`pip install -r requirement.txt`

### Train (finetuning)

For adapting for zero-shot adversarial robustness, run

`python -u ./EText.py`