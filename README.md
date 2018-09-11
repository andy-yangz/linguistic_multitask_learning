# Linguistic multi-task learning model 

Based on the code from of [jPTDP v1.0](https://github.com/datquocnguyen/jPTDP/releases), we add more flexibility and many improvement.

The basic model as follow:

![image.png](https://upload-images.jianshu.io/upload_images/4787675-64593d87cca9723a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

It jointly train [graph-based dependency parsing](http://www.coli.uni-saarland.de/~yzhang/rapt-ws1213/slides/valeeva.pdf) and POS tagging. Both of them can adjust the training layers. For embedding we concatenate char embedding (summarized by BiLSTM) and word embedding.

Later, we input predicted POS tag to higher task, as follow:

![image.png](https://upload-images.jianshu.io/upload_images/4787675-16f5a47b9877c81a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

In this situation, we have one more  embedding table for POS tags. This improved the performance indeed.

Later, we try to combine the morphlogical tagging task, the model like below:

![image.png](https://upload-images.jianshu.io/upload_images/4787675-7db7789d10cbd335.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

However, for this model, the performance not improved than the former model. So our best model is the above model, which has one additional POS tag embedding.

The master branch of this repository is the second model. Many good parameter setting I already set it into default. For morphological tagging, you can check morph branch.

### Installation

jPTDP requires the following software packages:

* `Python 2.7`
* [`DyNet` version 2.0](http://dynet.readthedocs.io/en/latest/python.html)

### Training model 

We train model upon [Universal Dependencies project](http://universaldependencies.org/). The data from this project follow 10 columns [CoNLL-U format](http://universaldependencies.org/format.html). We use the UPOS, FEATS, HEAD, and DEPREL as our primary training data.

The main hyper-parameters as follow:

 * `--dynet-mem`: Specify DyNet memory in MB, normally you don't need set.
 * `--pre_wembed`: Specify the pretrained word embedding directory.
 * `--epochs`: Specify number of traning epochs. Default value is 30.
 * `—model`: Specify a  name for model parameters file. Default value is "model".
 * `--wembedding`: Specify size of word embeddings. Default value is 100.
 * `--cembedding`: Specify size of character embeddings. Default value is 64.
 * `--pembedding`: Specify size of POS tag embeddings. Default value is 32.
 * `--pos_layer`: Specify number of POS tagging task layer. Default value is 1.
 * `--dep_layer`: Specify number of dependency parsing task layer. Default value is 1.
 * `--pos_lstm_dims`: Specify hidden dimensions for POS tagging task RNN. Default value is 2.
 * `--dep_lstm_dims`: Specify hidden dimensions for dependency parsing task RNN. Default value is 2.
 * `--arc_hidden`: Specify hidden dimensions for linear layer of arc prediction for dependency parsing. Default value is 100.
 * `--rel_hidden`: Specify hidden dimensions for linear layer of relation prediction for dependency parsing. Default value is 100.
 * `--params`: Specify a name for model hyper-parameters file. Default value is "model.params".
 * `--outdir`: Specify path to directory where the trained model will be saved. 
 * `--train`: Specify path to training data file.
 * `--dev`: Specify path to development data file. 

### Testing

* `--model`: Specify path to model parameters file.
* `--params`: Specify path to model hyper-parameters file.
* `--predict`: Specify the test mode.
* `--test`: Specify path to test file.
* `--outdir`: Specify path to directory where output file will be saved.
* `--output`: Specify name of the  output file.

