import argparse
from utils import levenshtein,get_vocab
import pdb
import re


def clean(label):
    alphabet = [a for a in '0123456789abcdefghijklmnopqrstuvwxyz* ']
    label = label.replace('-', '*')
    nlabel = ""
    for each in label.lower():
        if each in alphabet:
            nlabel += each
    return nlabel


parser = argparse.ArgumentParser()
parser.add_argument('--preds', type=str, default='../misc/preds/temp.txt', help='path to preds file')
parser.add_argument('--vocab', type=str, required=True,help="Path to desired vocabulary such as train/val/test")
parser.add_argument("--labels",type=str,required=True,help="Path to file containing all words")
parser.add_argument('--mode', type=str, default='word', help='path to preds file')
parser.add_argument('--lower', action='store_true', help='convert strings to lowercase before comparison')
parser.add_argument('--alnum', action='store_true', help='convert strings to alphanumeric before comparison')
opt = parser.parse_args()

train_vocab = get_vocab(opt.vocab,opt.labels)
# with open(opt.vocab) as f:
#     for line in f:
#         train_vocab.append(line.strip())


f = open(opt.preds, 'r')

tw = 0
ww = 0
tc = 0
wc = 0

word_lens = []
if opt.mode == 'word':
    for i , line in enumerate(f):
        print(line)
        if i%2==0:
            pred = line.strip()
        else:
            gt = line.strip()
            if gt in train_vocab:
                continue
            if opt.lower:
                gt = gt.lower()
                pred = pred.lower()
            if opt.alnum:
                pattern = re.compile('[\W_]+')
                gt = pattern.sub('', gt)
                pred = pattern.sub('', pred)
                # pdb.set_trace()
                # gt =
            # print('before')
            if gt != pred:
                ww += 1
                wc += levenshtein(gt, pred)
                word_lens.append(len(gt))
                print(f"GROUND TRUTH: {gt}, PREDICTION: {pred},WORD CHARACTER EDIT DISTANCE: {wc}")
            tc += len(gt)
            tw += 1
else:
    for i , line in enumerate(f):
        if i%2==0:
            pred = line.strip()
        else:
            gt = line.strip()
            gt = clean(gt)
            pred = clean(pred)
            gt_w = gt.split()
            pred_w = pred.split()
            for j in range(len(gt_w)):
                try:
                    if gt_w[j] != pred_w[j]:
                        # print(gt_w[j], pred_w[j])
                        ww += 1
                except IndexError:
                    ww += 1

            tw += len(gt.split())
            wc += levenshtein(gt, pred)
            tc += len(gt)


print(f"Wrong words: {ww}, Total words: {tw}")
print('WER: ', (ww/tw)*100)
print('CER: ', (wc/tc)*100)
print('Incorrect Avg: ', sum(word_lens)/len(word_lens))
print('Incorrect Max Avg: ', max(word_lens))
print('Incorrect Min Avg: ', min(word_lens))
