# coding=utf-8
from dynet import *
import dynet
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
from mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor


class jPosDepLearner:
    def __init__(self, vocab, pos, rels, morphs, w2i, c2i, options):
        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)
        #self.trainer = SimpleSGDTrainer(self.model)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.mdims = options.membedding_dims
        self.pdims = options.pembedding_dims
        self.pos_layer = options.pos_layer
        self.dep_layer = options.dep_layer

        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for ind, word in enumerate(pos)}
        self.morphs = {feats : ind for ind, feats in enumerate(morphs)} #
        self.id2morph = list(morphs)        
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels


        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        if self.bibiFlag:
            self.pos_builder = [VanillaLSTMBuilder(self.pos_layer, self.wdims + self.edim + self.cdims * 2, self.ldims, self.model),
                                VanillaLSTMBuilder(self.pos_layer, self.wdims + self.edim + self.cdims * 2, self.ldims, self.model)]
            self.dep_builders = [VanillaLSTMBuilder(self.dep_layer, self.ldims * 2 + self.pdims, self.ldims, self.model),
                                 VanillaLSTMBuilder(self.dep_layer, self.ldims * 2 + self.pdims, self.ldims, self.model)]



        self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims*2, len(self.pos), softmax))    

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        
        self.mlookup = self.model.add_lookup_parameters((len(morphs), self.mdims))
        self.plookup = self.model.add_lookup_parameters((len(pos), self.pdims))

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidBias = self.model.add_parameters((self.hidden_units))

            self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

            self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = self.model.add_parameters((len(self.irels)))
        
        self.char_rnn = RNNSequencePredictor(LSTMBuilder(1, self.cdims, self.cdims, self.model))
        self.charSeqPredictor = FFSequencePredictor(Layer(self.model, self.cdims*2, len(self.morphs), softmax))    


    def  __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())) # + self.outBias
        else:
            output = self.outLayer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()) # + self.outBias

        return output


    def __evaluate(self, sentence, train):
        exprs = [ [self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        return scores, exprs

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output.value(), output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.c2i)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                    evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)))] if self.external_embedding is not None else None
                    
                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[-1]

                    # char_state = dynet.noise(concatenate([last_state, rev_last_state]), 0.2)
                    # morph_logit = self.charSeqPredictor.predict_sequence(char_state)
                    # morphID = self.morphs.get(entry.feats)
                    # morphErrs.append(self.pick_neg_log(morph_logit, morphID))
                    # morph_emb = None
                    # for i in morph_logit:
                    #     morph_emb += i * self.mlookup(i)
                      
                    entry.vec = concatenate(filter(None, [wordvec, evec, last_state, rev_last_state]))
                    entry.ch_vec = concatenate([dynet.noise(fe,0.2) for fe in filter(None, [last_state, rev_last_state])])
                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:

                    morcat_layer = [entry.ch_vec for entry in conll_sentence]
                    morph_logits = self.charSeqPredictor.predict_sequence(morcat_layer)
                    predicted_morph_idx = [np.argmax(o.value()) for o in morph_logits]
                    predicted_morphs = [self.id2morph[idx] for idx in predicted_morph_idx]

                    lstm_forward = self.pos_builder[0].initial_state()
                    lstm_backward = self.pos_builder[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    pos_embed = []
                    concat_layer = [concatenate(entry.lstms) for entry in conll_sentence]
                    outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                    predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]  
                    predicted_postags = [self.id2pos[idx] for idx in predicted_pos_indices]
                    for pred in outputFFlayer:
                        pos_embed.append(soft_embed(pred.value(), self.plookup))
                
                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.dep_builders[0].initial_state()
                        blstm_backward = self.dep_builders[1].initial_state()

                        for entry, rentry, pembed, revpembed in zip(conll_sentence, reversed(conll_sentence),
                                                                    pos_embed, reversed(pos_embed)):
                            blstm_forward = blstm_forward.add_input(concatenate([entry.vec, pembed]))
                            blstm_backward = blstm_backward.add_input(concatenate([rentry.vec, revpembed]))

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                heads = decoder.parse_proj(scores)
                
                #Multiple roots: heading to the previous "rooted" one
                rootCount = 0
                rootWid = -1
                for index, head in enumerate(heads):
                    if head == 0:
                        rootCount += 1
                        if rootCount == 1:
                            rootWid = index
                        if rootCount > 1:    
                            heads[index] = rootWid
                            rootWid = index
                        
                
                for entry, head, pos, feats in zip(conll_sentence, heads, predicted_postags, predicted_morphs):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'
                    entry.pred_pos = pos
                    entry.pred_feats = feats

                dump = False

                if self.labelsFlag:
                    for modifier, head in enumerate(heads[1:]):
                        scores, exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence


    def Train(self, conll_path):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, self.c2i))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            posErrs = []
            morphErrs = []
            eeloss = 0.0

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 500 == 0 and iSentence != 0:
                    print "Processing sentence number: %d" % iSentence, ", Loss: %.2f" % (eloss / etotal), ", Time: %.2f" % (time.time()-start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                    evec = None

                    if self.external_embedding is not None:
                        evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                    #entry.vec = concatenate(filter(None, [wordvec, evec]))
                    
                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[-1]

                    # entry.vec = concatenate([dynet.noise(fe,0.2) for fe in filter(None, [wordvec, evec, last_state, rev_last_state])])
                    entry.vec = concatenate([dynet.noise(fe,0.2) for fe in filter(None, [wordvec, evec, last_state, rev_last_state])])
                    entry.ch_vec = concatenate([dynet.noise(fe,0.2) for fe in filter(None, [last_state, rev_last_state])])
                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    # Morphological layer

                    lstm_forward = self.pos_builder[0].initial_state()
                    lstm_backward = self.pos_builder[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)
                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    # POS layer
                    pos_embed = []
                    concat_layer = [concatenate(entry.lstms) for entry in conll_sentence]
                    concat_layer = [dynet.noise(fe,0.2) for fe in concat_layer]
                    outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                    posIDs  = [self.pos.get(entry.pos) for entry in conll_sentence ]
                    for pred, gold in zip(outputFFlayer, posIDs):
                        posErrs.append(self.pick_neg_log(pred,gold))
                    # POS embedding
                        pos_embed.append(soft_embed(pred.value(), self.plookup))    

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.dep_builders[0].initial_state()
                        blstm_backward = self.dep_builders[1].initial_state()

                        for entry, rentry, pembed, revpembed in zip(conll_sentence, reversed(conll_sentence),
                                                                    pos_embed, reversed(pos_embed)):
                            blstm_forward = blstm_forward.add_input(concatenate([entry.vec, pembed]))
                            blstm_backward = blstm_backward.add_input(concatenate([rentry.vec, revpembed]))

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    for modifier, head in enumerate(gold[1:]):
                        rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                        wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                        if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                            lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)
                

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0 or len(posErrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0 or len(posErrs) > 0:
                        eerrs = (esum(errs + lerrs + posErrs + morphErrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []
                        posErrs = []
                        morphErrs = []
                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs + posErrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            posErrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update()
        print "Loss: %.2f" % (mloss/iSentence)


def soft_embed(vec, lookup):
    embeds = []
    for i, v in enumerate(vec):
        embeds.append(v * lookup[i])
    return esum(embeds)