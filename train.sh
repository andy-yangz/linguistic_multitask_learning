python jPTDP.py --dynet-seed 123456789 --dynet-mem 1000 --epochs 30 --wembedding 128 --cembedding 64 --model trialmodel --params trialmodel.params --outdir sample/ --train sample/train.conllu --dev sample/dev.conllu

#Test
python jPTDP.py --dynet-seed 123456789 --dynet-mem 1000 --epochs 30 --model English --params English.params --outdir output/gold_morph/ --predict --test /home/andy/data/ud/ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/en.conllu
