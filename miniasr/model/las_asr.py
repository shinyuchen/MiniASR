def CreateOnehotVariable( input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x

# TimeDistributed function
# This is a pytorch version of TimeDistributed layer in Keras I wrote
# The goal is to apply same module on each timestep of every instance
# Input : module to be applied timestep-wise (e.g. nn.Linear)
#         3D input (sequencial) with shape [batch size, timestep, feature]
# output: Processed output      with shape [batch size, timestep, output feature dim of input module]
def TimeDistributed(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size,time_steps,-1)
import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np



# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        if(timestep % 2 != 0):
          input_x = torch.cat((input_x, torch.zeros((batch_size, 1, feature_dim)).cuda()), dim = 1)
          timestep += 1
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim=240, listener_hidden_dim=256, listener_layer=2, rnn_unit='LSTM', use_gpu=True, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.listener_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.listener_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        
        return output


# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim=31,  speller_hidden_dim=512, rnn_unit="LSTM", speller_rnn_layer=1, use_gpu=True, max_label_len=1700,
                 use_mlp_in_attention=True, mlp_dim_in_attention=128, mlp_activate_in_attention='relu', listener_hidden_dim=1024,
                 multi_head=4, decode_mode=1, **kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.rnn_layer = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.attention = Attention( mlp_preprocess_input=use_mlp_in_attention, preprocess_mlp_dim=mlp_dim_in_attention,
                                    activate=mlp_activate_in_attention, input_feature_dim=2*listener_hidden_dim,
                                    multi_head=multi_head)
        self.character_distribution = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_gpu:
            self = self.cuda()

    # Stepwise operation of each sequence
    def forward_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        attention_score, context = self.attention(rnn_output,listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))

        return raw_pred, hidden_state, context, attention_score

    def forward(self, listener_feature, ground_truth=None, teacher_force_rate = 0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]

        output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        if self.use_gpu:
            output_word = output_word.cuda()
        rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)
        print(rnn_input.size())
        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []

        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]

        for step in range(max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:,step:step+1,:].type(self.float_type)
            else:
                # Case 0. raw output as input
                if self.decode_mode == 0:
                    output_word = raw_pred.unsqueeze(1)
                # Case 1. Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(raw_pred.topk(1)[1]):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)             
                # Case 2. Sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(raw_pred).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(sampled_word):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                
            rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)

        return raw_pred_seq,attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, 1, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size,  listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):  

    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, mode='dot', input_feature_dim=512,
                multi_head=1):
        super(Attention,self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim*multi_head)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            if self.multi_head > 1:
                self.dim_reduce = nn.Linear(input_feature_dim*multi_head,input_feature_dim)
            if activate != 'None':
                self.activate = getattr(F,activate)
            else:
                self.activate = None

    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.psi,listener_feature))
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_listener_feature = TimeDistributed(self.psi,listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature

        if self.mode == 'dot':
            if self.multi_head == 1:
                energy = torch.bmm(comp_decoder_state,comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
                attention_score = [self.softmax(energy)]
                context = torch.sum(listener_feature*attention_score[0].unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)
            else:
                attention_score =  [ self.softmax(torch.bmm(att_querry,comp_listener_feature.transpose(1, 2)).squeeze(dim=1))\
                                    for att_querry in torch.split(comp_decoder_state, self.preprocess_mlp_dim, dim=-1)]
                projected_src = [torch.sum(listener_feature*att_s.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1) \
                                for att_s in attention_score]
                context = self.dim_reduce(torch.cat(projected_src,dim=-1))
        else:
            # TODO: other attention implementations
            pass
        
        

        return attention_score,context








def label_smoothing_loss(pred_y,true_y,label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y,dim=-1),dim=-1,keepdim=True)
    
    # calculate smoothen label, last term ensures padding vector remains all zero
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0-label_smoothing)*true_y+(label_smoothing/class_dim))*torch.sum(true_y,dim=-1,keepdim=True)

    loss = - torch.mean(torch.sum((torch.sum(smooth_y * pred_y,dim=-1)/seq_len),dim=-1))

    return loss

def batch_iterator(batch_data, batch_label, listener, speller, optimizer, tf_rate=0.9, is_training=True, max_label_len=1700, **kwargs):
    bucketing = False
    use_gpu = True
    output_class_dim = 31
    label_smoothing = 0.1

    # Load data
    if bucketing:
        batch_data = batch_data.squeeze(dim=0)
        batch_label = batch_label.squeeze(dim=0)
    current_batch_size = len(batch_data)
    max_label_len = min([batch_label.size()[1], max_label_len])

    batch_data = Variable(batch_data).type(torch.FloatTensor)
    batch_label = Variable(batch_label, requires_grad=False)
    criterion = nn.NLLLoss(ignore_index=0)
    if use_gpu:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        criterion = criterion.cuda()
    # Forwarding
    listner_feature = listener(batch_data)
    if is_training:
        raw_pred_seq, _ = speller(listner_feature,ground_truth=batch_label,teacher_force_rate=tf_rate)
    else:
        raw_pred_seq, _ = speller(listner_feature,ground_truth=None,teacher_force_rate=0)

    pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1)[:,:max_label_len,:]).contiguous()

    if label_smoothing == 0.0 or not(is_training):
        pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        true_y = torch.max(batch_label,dim=2)[1][:,:max_label_len].contiguous()#.view(-1)

        loss = criterion(pred_y,true_y)
        # variable -> numpy before sending into LER calculator
        # batch_ler = LetterErrorRate(torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
        #                             true_y.cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    else:
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y,true_y,label_smoothing=label_smoothing)
        # batch_ler = LetterErrorRate(torch.max(pred_y,dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
        #                             torch.max(true_y,dim=2)[1].cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    # if is_training:
    #     loss.backward()
    #     optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    


    return batch_loss #, batch_ler
'''
    File      [ ctc_asr.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ CTC ASR model. ]
'''

import logging
import numpy as np
import torch
from torch import nn

from miniasr.model.base_asr import BaseASR
from miniasr.module import RNNEncoder


class ASR(BaseASR):
    '''
        LAS ASR model
    '''

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

        # Main model setup
        self.Listen = Listener()

        self.AttendAndSpell = Speller()

        # Loss function (CTC loss)
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        # Beam decoding with Flashlight
        self.enable_beam_decode = False
        if self.args.mode in {'dev', 'test'} and self.args.decode.type == 'beam':
            self.enable_beam_decode = True
            self.setup_flashlight()

    def setup_flashlight(self):
        '''
            Setup flashlight for beam decoding.
        '''
        import math
        from flashlight.lib.text.dictionary import (
            Dictionary, load_words, create_word_dict)
        from flashlight.lib.text.decoder import (
            CriterionType, LexiconDecoderOptions, LexiconDecoder, KenLM, Trie, SmearingMode)

        token_dict = Dictionary(self.args.decode.token)
        lexicon = load_words(self.args.decode.lexicon)
        word_dict = create_word_dict(lexicon)

        lm = KenLM(self.args.decode.lm, word_dict)

        sil_idx = token_dict.get_index("|")
        unk_idx = word_dict.get_index("<unk>")

        trie = Trie(token_dict.index_size(), sil_idx)
        start_state = lm.start(False)

        for word, spellings in lexicon.items():
            usr_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, usr_idx)
            for spelling in spellings:
                # convert spelling string into vector of indices
                spelling_idxs = [token_dict.get_index(
                    token) for token in spelling]
                trie.insert(spelling_idxs, usr_idx, score)
        trie.smear(SmearingMode.MAX)

        options = LexiconDecoderOptions(
            self.args.decode.beam_size,
            self.args.decode.token_beam_size,
            self.args.decode.beam_threshold,
            self.args.decode.lm_weight,
            self.args.decode.word_score,
            -math.inf,
            self.args.decode.sil_score,
            self.args.decode.log_add,
            CriterionType.CTC
        )

        blank_idx = token_dict.get_index("#")  # for CTC
        is_token_lm = False  # we use word-level LM
        self.flashlight_decoder = LexiconDecoder(
            options, trie, lm, sil_idx, blank_idx, unk_idx, [], is_token_lm)
        self.token_dict = token_dict

        logging.info(
            f'Beam decoding with beam size {self.args.decode.beam_size}, '
            f'LM weight {self.args.decode.lm_weight}, '
            f'Word score {self.args.decode.word_score}')

    def forward(self, wave, wave_len, text, text_len):
        '''
            Forward function to compute logits.
            Input:
                wave [list]: list of waveform files
                wave_len [long tensor]: waveform lengths
            Output:
                logtis [float tensor]: Batch x Time x Vocabs
                enc_len [long tensor]: encoded length (logits' lengths)
                feat [float tensor]: extracted features
                feat_len [long tensor]: length of extracted features
        '''

        # Extract features
        feat, feat_len = self.extract_features(wave, wave_len)

        # Encode features
        loss = batch_iterator(feat, text, self.Listen, self.AttendAndSpell, self.optimizers())

        # Project hidden features to vocabularies
        

        return loss

    def training_step(self, batch, batch_idx):
        ''' Processes in a single training loop. '''

        # Get inputs from batch
        wave, text = batch['wave'], batch['text']
        wave_len, text_len = batch['wave_len'], batch['text_len']

        # Compute logits
        loss = self(wave, wave_len, text, text_len)

        # Compute loss
        # loss = self.cal_loss(logits, enc_len, feat, feat_len, text, text_len)

        # Log information
        self.log('train_loss', loss)

        return loss


    def decode(self, logits, enc_len, decode_type=None):
        ''' Decoding. '''
        if self.enable_beam_decode and decode_type != 'greedy':
            return self.beam_decode(logits, enc_len)
        return self.greedy_decode(logits, enc_len)
    def validation_step(self, batch, batch_idx):
      pass
    def validation_epoch_end(self, outputs):
      pass
    def greedy_decode(self, logits, enc_len):
        ''' CTC greedy decoding. '''
        hyps = torch.argmax(logits, dim=2).cpu().tolist()  # Batch x Time
        return [self.tokenizer.decode(h[:enc_len[i]], ignore_repeat=True)
                for i, h in enumerate(hyps)]

    def beam_decode(self, logits, enc_len):
        ''' Flashlight beam decoding. '''

        greedy_hyps = self.greedy_decode(logits, enc_len)
        log_probs = torch.log_softmax(logits, dim=2) / np.log(10)

        beam_hyps = []
        for i, log_prob in enumerate(log_probs):
            emissions = log_prob.cpu()
            hyps = self.flashlight_decoder.decode(
                emissions.data_ptr(), enc_len[i], self.vocab_size)

            if len(hyps) > 0 and hyps[0].score < 10000.0:
                hyp = self.tokenizer.decode(hyps[0].tokens, ignore_repeat=True)
                beam_hyps.append(hyp.strip())
            else:
                beam_hyps.append(greedy_hyps[i])

        return beam_hyps
