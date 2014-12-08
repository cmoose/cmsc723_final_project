#!/bin/python
import os
import pickle
import sys
import re
from util import *

DIR_PREFIX = './wikipedia/'

#This creates a separate set of files with just the info we want
def preprocess_articles():
    for d in os.listdir(DIR_PREFIX + 'wikipedia/'):
        for f in os.listdir(DIR_PREFIX + 'wikipedia/' + d):
            cmd = 'cat "%s/wikipedia/%s/%s" | grep NN | awk \'{print $3}\' | sort | uniq -c | sort -n > "%s/wordcount/%s/%s"' \
                % (DIR_PREFIX,d,f,DIR_PREFIX,d,f)
            ret = os.system(cmd)
            print ret, f

#Builds pickle files from preprocessing
def build_pickle_files():
    commapickfh = open(DIR_PREFIX + 'comma.pkl', 'wb')
    comma_articles = []
    for d in os.listdir(DIR_PREFIX + 'wordcount'):
        pickfh = open(DIR_PREFIX + d + '.pkl', 'wb')
        docs = {}
        for f in os.listdir(DIR_PREFIX + 'wordcount/' + d):
            #Some of the answers have truncated article names, 
            #they are ones with commas in the title
            if re.search(',', f):
                comma_articles.append(f)
            fh = open(DIR_PREFIX + 'wordcount/%s/%s' % (d,f))
            docs[f] = {}
            for line in fh.readlines():
                doc = Counter()
                count = line.split()[0].strip()
                word = line.split()[1].strip()
                docs[f][word] = count
        print "Creating pickle %s file" % (d + '.pkl')
        pickle.dump(docs,pickfh)
        pickfh.close()
    pickle.dump(comma_articles,commapickfh)

def load_pickle_files():
    all_docs = {}
    print 'Loading Wikipedia...'
    for d in os.listdir(DIR_PREFIX + 'wordcount'):
        print 'load %s articles.' % (d)
        pickfh = open(DIR_PREFIX + d + '.pkl')
        docs = pickle.load(pickfh)
        all_docs[d] = docs
    return all_docs

def load_comma_pickle_file():
    comma_docs = []
    print 'Loading comma articles.'
    pickfh = open(DIR_PREFIX + 'comma.pkl')
    comma_docs = pickle.load(pickfh)
    return comma_docs

if __name__ == '__main__':
    pass
