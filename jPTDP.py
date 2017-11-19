# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time
from collections import defaultdict


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--pre_wembed", dest="pretrain_wembed", help="Pretrained Word embeddings", metavar="FILE")    
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=64)
    parser.add_option("--membedding", type="int", dest="membedding_dims", default=64)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=32)
    parser.add_option("--pos_layer", type="int", dest="pos_layer", default=1)
    parser.add_option("--dep_layer", type="int", dest="dep_layer", default=1)
    parser.add_option("--pos_dropout", type="float", dest="pos_dropout", default=0.2)
    parser.add_option("--dep_dropout", type="float", dest="dep_dropout", default=0.2)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--arc_hidden", type="int", dest="arc_hidden", default=100)
    parser.add_option("--rel_hidden", type="int", dest="rel_hidden", default=100)    
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    # parser.add_option("--lr", type="float", dest="learning_rate", default=0.001)
    parser.add_option("--outdir", type="string", dest="outdir", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--rnn_type", type="string", dest="rnn_type", default="LSTM")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--pos_lstm_dims", type="int", dest="pos_lstm_dims", default=128)
    parser.add_option("--dep_lstm_dims", type="int", dest="dep_lstm_dims", default=128)
    parser.add_option("--gold_pos", action="store_true", dest="gold_pos", default=False)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=123456789)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        with open(os.path.join(options.outdir, options.params), 'r') as paramsfp:
            words, w2i, c2i, pos, rels, morphs, stored_opt = pickle.load(paramsfp)
            
        stored_opt.external_embedding = options.external_embedding
        
        print 'Loading pre-trained joint model'
        parser = learner.jPosDepLearner(words, pos, rels, morphs, w2i, c2i, stored_opt)
        parser.Load(os.path.join(options.outdir, os.path.basename(options.model)))
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        tespath = os.path.join(options.outdir, 'test_pred.conll' if not conllu else 'test_pred.conllu')
        print 'Predicting POS tags and parsing dependencies'
        devPredSents = parser.Predict(options.conll_test)

        count = 0
        uasCount = 0
        lasCount = 0
        posCount = 0
        morphCount = 0
        poslasCount = 0

        for idSent, devSent in enumerate(devPredSents):
            conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
            sent = ' '.join([entry.form for entry in conll_devSent if entry.id > 0])
            for entry in conll_devSent:
                if entry.id <= 0:
                    continue
                if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                    poslasCount += 1
                if entry.pos == entry.pred_pos:
                    posCount += 1
                if entry.feats == entry.pred_feats:
                    morphCount += 1
                if entry.parent_id == entry.pred_parent_id:
                    uasCount += 1
                if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                    lasCount += 1
                count += 1
       
        print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
        print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
        print "POS accuracy:\t%.2f" % (float(posCount) * 100 / count)
        print "Morph accuracy:\t%.2f" % (float(morphCount) * 100 / count)
        print "POS&LAS:\t%.2f" % (float(poslasCount) * 100 / count)

        # ts = time.time()
        # test_res = list(devPredSents)
        # te = time.time()
        # print 'Finished in', te-ts, 'seconds.'
        # utils.write_conll(tespath, test_res)

        # if not conllu:#Scored with punctuation
        #    os.system('perl utils/eval07.pl -q -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.scores.txt')
        # else:
        #    os.system('python utils/evaluation_script/conll17_ud_eval.py -v -w utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + tespath + ' > ' + tespath + '.scores.txt')
    else:
        print 'Extracting vocabulary'
        words, w2i, c2i, pos, rels, morphs = utils.vocab(options.conll_train)
        
        with open(os.path.join(options.outdir, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, c2i, pos, rels, morphs, options), paramsfp)
        
        print 'Initializing joint model'
        print 'RNN type: ' + options.rnn_type
        print 'POS layer: %d, POS LSTM dims: %d' % (options.pos_layer, options.pos_lstm_dims)
        print 'Dep layer: %d, Dep LSTM dims: %d' % (options.dep_layer, options.dep_lstm_dims)
        # print 'Learning Rate: %f' % (options.learning_rate)
        parser = learner.jPosDepLearner(words, pos, rels, morphs, w2i, c2i, options)
        
        highestScore = 0.0
        eId = 0
        for epoch in xrange(options.epochs):
            print '\n-----------------\nStarting epoch', epoch + 1
            parser.Train(options.conll_train)
            
            if options.conll_dev == "N/A":  
                parser.Save(os.path.join(options.outdir, os.path.basename(options.model)))
                
            else: 
                devPredSents = parser.Predict(options.conll_dev)
                
                count = 0
                uasCount = 0
                lasCount = 0
                posCount = 0
                morphCount = 0
                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
                    
                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            poslasCount += 1
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                        if entry.feats == entry.pred_feats:
                            morphCount += 1
                        if entry.parent_id == entry.pred_parent_id:
                            uasCount += 1
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        count += 1
                        
                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
                print "POS accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "Morph accuracy:\t%.2f" % (float(morphCount) * 100 / count)
                print "POS&LAS:\t%.2f" % (float(poslasCount) * 100 / count)
                
                score = float(poslasCount) * 100 / count
                if score >= highestScore:
                    parser.Save(os.path.join(options.outdir, os.path.basename(options.model)))
                    highestScore = score
                    eId = epoch + 1
                
                print "Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId)