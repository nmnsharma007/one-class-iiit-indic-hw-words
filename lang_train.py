from base_train import *
from config import *

from torchvision.transforms import Compose, ColorJitter, RandomApply
from datasets.imprint_dataset import Rescale as IRescale
from datasets.grid_distort import GD

import warnings
warnings.filterwarnings("ignore")

opt = parser.parse_args()

lang = f'{opt.lang}'
dataset_name = f'{opt.lang}'
opt.alphabet_type = "file"
opt.alphabet = f"{opt.lang}.txt"
opt.trainRoot = f'/data/iiit-indic-hw-words/datasets/{opt.lang}/iiit-indic/{opt.lang}-train-lmdb'
opt.labels = f'/data/iiit-indic-hw-words/datasets/{opt.lang}/iiit-indic/vocab.txt'

if opt.mode == 'test':
    opt.valRoot = f'/data/iiit-indic-hw-words/datasets/{opt.lang}/iiit-indic/{opt.lang}-test-lmdb'
    print(f'Testing on {opt.valRoot}')
else:
	opt.valRoot = f'/data/iiit-indic-hw-words/datasets/{opt.lang}/iiit-indic/{opt.lang}-val-lmdb'

opt.valInterval = 1000
opt.displayInterval = 100
opt.nepoch = 5
opt.cuda = True
opt.random_sample = True
opt.workers = 10
opt.batchSize = 64
opt.ngpu = 1


opt.adadelta = True
opt.lr = 0.001
# opt.STN_type = 'Affine'
# opt.tps_inputsize = [32, 64]

opt.STN_type = 'TPS'
opt.tps_inputsize = [48, 128]
opt.tps_outputsize = [96, 256]

htr = BaseHTR(opt, dataset_name)
htr.nheads = 1

l1 = ['pn', 'bn', 'od', 'gu'] # indo-aryan languages smaller width
l2 = ['kn', 'ma', 'ta']
if lang in l1:
    elastic_alpha = 0.3
else:
    elastic_alpha = 0.2
htr.train_transforms = Compose([
								GD(0.5),
								IRescale(max_width=htr.opt.imgW, height=htr.opt.imgH),
								ElasticTransformation(0.5, alpha=elastic_alpha),
								AffineTransformation(0.5, rotate=5, shear=0.5),
								RandomApply([ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
	                            ToTensor()])

htr.test_transforms = Compose([IRescale(max_width=htr.opt.imgW, height=htr.opt.imgH),
                               ToTensor()])
htr.run()
