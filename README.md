# Project Code for NLP Course at UVA (2023)

## How to run?
- Unpack FARN dataset files to the same directory: Fake.csv.zip and True.csv.zip.
- Run each python file on a GPU device. Remember to modify CUDA device in the code and hyper-parameters used in the code.

```python
python3 switch_transformer.py  # on FARN dataset
python3 bert.py  # on FARN dataset
python3 switch_transformer_liar.py  # on Liar dataset
python3 bert_liar.py  # on Liar dataset
```
