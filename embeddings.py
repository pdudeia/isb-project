import nltk
import pandas as pd
import os
import pickle
import re
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing
import gensim
import json
import gc
import progressbar
import warnings
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

def update_progress(progress):
    import sys
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "="*block + " "*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.losses.append(loss)
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.losses.append(loss-self.loss_previous_step)
        self.epoch += 1
        self.loss_previous_step = loss

def unique(list_):
    return list(set(list_))
def intersect(list_lof_lists):
    return list(set.intersection(*[set(x) for x in list_lof_lists]))

def get_similar_words(text,model,look_up_list,refrence_word = None):
    ## unique words from report
    words = [x for line in text for x in line]
    words = unique(words)
    ## similar words from the model and lookup list ##
    list_words = []
    for v in look_up_list:
        if refrence_word is None:
            similar_words = [x[0] for x in model.wv.most_similar(positive = v, topn = 50)]
        if refrence_word is not None:
            similar_words = [x[0] for x in model.wv.most_similar(positive = [v,refrence_word], topn = 50)]
        similar_words = intersect([similar_words,words])
        list_words.append(similar_words)
    ## flatten list
    list_words = [x for word in list_words for x in word]
    return unique(list_words)

# model = KeyedVectors.load_word2vec_format('/home/31265/10K_Pandemic/glove.6B.300d.w2vformat.txt')
#model = gensim.models.Word2Vec.load('/home/31265/10K_Pandemic/who_disease_outbreak.model')

def model_build(sentences, cobj,window=5,min_count=5,negative=5,sample=1e-3,epoch=1,alpha=0.025):
    cores = multiprocessing.cpu_count()

    model_2 = gensim.models.Word2Vec(size=300, window=window, min_count=min_count, workers=cores-1, negative=negative,sample=sample,alpha=alpha)
    model_2.build_vocab(sentences)
    total_examples = model_2.corpus_count
    print(f'Total examples : {total_examples}')

    # model_2.build_vocab([list(model.wv.vocab.keys())], update=True)
    # model_2.intersect_word2vec_format('/home/31265/10K_Pandemic/glove.6B.300d.w2vformat.txt', binary=False, lockf=1.0)

    model_2.train(sentences, total_examples=total_examples, epochs=epoch, compute_loss=True, callbacks=[cobj])

    return model_2

def search_results(text,model,search_dict):
    words = {}
    for key,value in search_dict.items():
        if key == 'None':
            x = get_similar_words(text,model,value[1])
            words[value[0]] = x
        if key != 'None':
            x = get_similar_words(text,model,value[1],refrence_word=key)
            words[str(value[0])+'_'+str(key)] = x
    return words

def main():
    ## key words to find ##
    in_path = 'text_data/'
    out_path = 'models/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    files = glob(in_path+'*.txt')
    # params = {
    #         2008:{'x_alpha': 0.00147651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2009:{'x_alpha': 0.00147651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2010:{'x_alpha': 0.00147651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2011:{'x_alpha': 0.00147651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2010:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2012:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2013:{'x_alpha': 0.0017651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.0064077618550642e-05, 'x_window': 2.0},
    #         2014:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         2015:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         2016:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         # 2016:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         2017:{'x_alpha': 0.00147651091635882, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         2018:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0},
    #         2019:{'x_alpha': 0.0018590758976614816, 'x_epoch': 100.0, 'x_min_count': 3.0, 'x_negative': 5.0, 'x_sample': 1.00e-05, 'x_window': 2.0}}

    for file in [files[3]]:
        best_params = {'x_alpha': 1e-6, 'x_epoch': 500, 'x_min_count': 1, 'x_negative': 10, 'x_sample': 0, 'x_window': 5}
        # best_params['x_epoch'] = 500
        print(file, best_params)
        with open(file) as f:
            text = list(filter(None, f.read().split('\n')))
            f.close()

        text = [[x.casefold() for x in line.split(' ')] for line in text]
        cobj = callback()
        test_model = model_build(text, cobj, window=int(best_params['x_window']), min_count=int(best_params['x_min_count']), negative=int(best_params['x_negative']),
                                sample=best_params['x_sample'], epoch=int(best_params['x_epoch']), alpha=best_params['x_alpha'])
        # test_model.wv.save_word2vec_format(out_path+file.split('\\')[-1].split(".")[0]+'.model')
        # test_model.wv.save_word2vec_format(out_path+file.split('\\')[-1].split(".")[0]+'.txt')
        print('-'*50)
        print(test_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv')))
        print(test_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))[0])
        print('-'*50)
        fig, axis = plt.subplots(1,1,figsize=(15,10))
        axis.set_xlabel('Epochs')
        axis.set_ylabel('Loss')
        axis.plot(cobj.losses, color='blue')
        plt.show()
        break
        del test_model
        gc.collect()
    return

if __name__ == '__main__':
    main()
