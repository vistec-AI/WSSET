# Self-supervised Deep Metric Learning for Pointsets
This is a TF2 implementation of the paper: Self-supervised Deep Metric Learning for Pointsets, ICDE 2021 and the extended version: Towards Pointsets Representation Learning via Self-Supervised Learning and Set Augmentation, TPAMI. We proposed a novel deep representation learning technique for pointset using zero training labels called "Weighted Self-supervised EMD Triplet loss." You can find the link of this paper here. [[ICDE version]](https://personal.ntu.edu.sg/c.long/paper/21-ICDE-pointSet.pdf) [[TPAMI version]](https://personal.ntu.edu.sg/c.long/paper/22-TPAMI-pointSet.pdf)

## Dataset preparation
We follow Matthew J Kusner's paper "From Word Embeddings to Document Distances" by using set of word vectors dataset provided in [this repository](https://github.com/mkusner/wmd)
and placing `.mat` files into `data/` folder. They also provided codes for converting raw text data into sets of word vectors too.

Also, we need to calculate pairwise EMD for each sample in the dataset using Python EMD module provided in [the same repository](https://github.com/mkusner/wmd/tree/master/python-emd-master) by converting the output in `.pk` into `.csv` and placing it inside `data/wmd/`.

In this repository, we provided BBCSports dataset for both pointset data and calculated EMD as the example.


## Requirements
* Python 3.8
* pip

You can create a new environment and install required packages using:
```
pip install -r requirements.txt
```


## Pretraining a based model using Weighted Self-supervised EMD Triplet loss (WSSET)
```
python train.py --dsname bbcsport --batchsize 32 --lr 1e-4 --epoch 1000  --loss gaussian --treshold 5
```
* To use Weighted Self-supervised EMD Triplet loss for training, please use `--loss gaussian`.
* To use approximate EMD loss, please use `--loss `.

## Fine-tuning a pretrained model
```
python train.py --dsname bbcsport --batchsize 32 --lr 1e-4 --epoch 1000  --loss triplet --treshold 5 --istransfer --loadmodel pretrained_bbcsport.h5
```
You can set the flag to train in fine-tuning step by `--istransfer` and specify a directory of the pretrained model using `--loadmodel directory\model.h5`.
We fine-tune a model using `--loss ce` (crossentropy loss) and `--loss triplet` (semi-hard negative mining triplet loss), which require labels.

## Cite and contact
pattaramanee dot a underscore s19 at vistec dot ac dot th 
```
@INPROCEEDINGS{WSSET
  author={P. {Arsomngern} and C. {Long} and S. {Suwajanakorn} and S. {Nutanong}},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)}, 
  title={Self-Supervised Deep Metric Learning for Pointsets}, 
  year={2021}}
```
```
@INPROCEEDINGS{
@ARTICLE{9665285,
  author={Arsomngern, Pattaramanee and Long, Cheng and Suwajanakorn, Supasorn and Nutanong, Sarana},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Pointsets Representation Learning via Self-Supervised Learning and Set Augmentation}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3139113}}
```
## Reference code
We use Keras implementation of self-attention module from [CyberZHG repository](https://github.com/CyberZHG/keras-transformer) and modifying semi-hard negative mining triplet loss from [Tensorflow Addons repository](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py).
