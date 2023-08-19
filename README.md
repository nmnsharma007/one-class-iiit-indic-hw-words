# Indic-HTR
Handwriting recognition for various Indic scripts using deep learning(STN, CNN, LSTM)

For more info about the pipeline, please refer to *IIIT-INDIC-HW-WORDS: A Dataset for Indic Handwritten Text Recognition*

[\[Paper\]](http://cvit.iiit.ac.in/images/ConferencePapers/2021/iiit-indic-hw-words.pdf)| [\[Dataset\]](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words) | [\[Teaser\]](http://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/331.mp4)

## Training and Evaluation:

### Dataset preparation:
- Create LMDB files for train, validation and test splits.
```
python tools/create_dataset.py --root_dir <dataset_dir> --save <lmdb_dst_path> --lang <lang_code>
```
The dataset folder should follow the same structure as IIIT-INDIC-HW-WORDS structure.

- Generate a file containing Unicode symbols/characters to be used for prediction. Move this file to alphabet/ folder.
  This repo already contains the sorted alphabet list for Indic scripts in the alphabet/ folder.

## Training:

To train model(TPS-ResNet-BiLSTM-CTC) from scratch:
```
python lang_train.py --mode train --lang <lang_code> --trainRoot <train_lmdb_path> --valRoot <val_lmdb_path> --cuda
```
Refer to *lang_train.py* and *config.py* for default settings and additional parameter settings.

Language codes for Indic scripts:
|Bengali|Gujarati|Gurumukhi|Odia|Kannada|Malayalam|Tamil|Urdu|
|-------|--------|---------|----|-------|---------|-----|----|
|   bn  |   gu   |    pn   | od |   kn  |    ma   |  ta | ur |

## Evaluation and testing:
To generate predictions for a <test-lmdb> file, try:
```
python lang_train.py --lang <lang_code> --mode test --valRoot <test-lmdb-path> --pretrained <trained_model_path> --out <save-predictions-path> 
```
To evaluate the generated predictions, try the following:
```
python tools/score.py --preds <save-predictions-path>
```
or
```
python tools/oov_score.py --preds <save-predictions-path> --vocab <path-to-train-vocab> --labels <path-to-all-labels-file>
```
to get WER and CER for OOV words only.

```
