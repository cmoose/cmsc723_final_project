import csv
import numpy
import pickle
import os
from util import *
import re

PKL_TRAIN = 'data/train.pkl'
PKL_DEV = 'data/dev.pkl'
TRAIN = 'data/train.csv'
VW_INPUT_TRAIN = 'vw/qa_vw.tr'
VW_INPUT_DEV = 'vw/qa_vw.de'
VW_MODEL = 'vw/qa_vw.model'
vw_bin = '/usr/local/bin/vw'

def main():
    #answer_map = [] #Index is the map
    if (not os.path.isfile(PKL_TRAIN)) | (not os.path.isfile(PKL_DEV)):
        train = load_training_data(TRAIN)
        generate_train_dev(train)

    train = pickle.load(open(PKL_TRAIN, "rb"))
    dev = pickle.load(open(PKL_DEV, "rb"))
    #answer_map = create_answer_map(train)

    print 'Generating Classification Data'
    generateVWData(dev, VW_INPUT_DEV)
    generateVWData(train, VW_INPUT_TRAIN)
    
    trainVW(VW_INPUT_DEV, VW_MODEL, True)
    
    train_pred = testVW(VW_INPUT_DEV, VW_MODEL, True)
    
    print len(train)
    print len(dev)


def load_training_data(filename):
    data = {}
    with open(filename, 'rb') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in content:
            question_id = row[0]
            if question_id not in 'Question ID':
                text = row[1]
                quanta = format_scores(row[2].split(','))
                answer = row[3]
                sentence_pos = row[4]
                wiki_score = format_scores(row[5].split(','))
                category = row[6]
                new_line = {'id': question_id, 'text': text, 'quanta': quanta, 'answer': answer,
                            'sentence_pos': sentence_pos,
                            'wiki_score': wiki_score, 'category': category}
                if data.get(question_id) is None:
                    data[question_id] = []
                data[question_id].append(new_line)
    return data


def format_scores(data):
    formatted_data = {}
    for item in data:
        split_data = item.split(':')
        if len(split_data) == 2:
            answer = split_data[0].strip()
            confidence = float(split_data[1])
            formatted_data[confidence] = answer
    return formatted_data


def create_answer_map(data):
    answer_map = []
    for item in data:
        for k,v in item['wiki_score'].items():
            if answer_map.count(v) == 0:
                answer_map.append(v)
        for k,v in item['quanta'].items():
            if answer_map.count(v) == 0:
                answer_map.append(v)
    return answer_map

def generate_train_dev(data):
    train, dev = set_dev_data(data)
    pickle.dump(train, open(PKL_TRAIN, 'wb'))
    pickle.dump(dev, open(PKL_DEV, 'wb'))
    print 'The size of the training set is: ', len(train)
    print 'The size of the dev set is: ', len(dev)
    return train, dev


def set_dev_data(data):
    selected_data = []
    for item in data.values():
        # The data is a collection of questions,
        # we don't want to allow duplicate questions,
        # here we select the first object for the question,
        # in this case it gives us the easiest to predict test set
        selected_data.append(item[len(item)-1])
    numpy.random.shuffle(selected_data)
    split = int(len(selected_data) * .80)
    train = selected_data[:split]
    dev = selected_data[split + 1:]
    return train, dev

def answerFeatures(item):
    array_of_answers = []
    correct_answer = item['answer']
    for k,v in item['wiki_score'].items():
        feats = Counter()
        feats['a_' + v] = 1
        feats['wiki_prob'] = k
        
        isCorrect = 1 #False
        if v == correct_answer:
            isCorrect = 0
        
        #append new answer to array_of_answers
        new_answer = (isCorrect, feats, {})
        array_of_answers.append(new_answer)
        
    for k,v in item['quanta'].items():
        feats = Counter()
        feats['a_' + v] = 1
        feats['quanta_prob'] = k

        isCorrect = 1 #False
        if v == correct_answer:
            isCorrect = 0
        
        #append new answer to array_of_answers
        new_answer = (isCorrect, feats, {})
        array_of_answers.append(new_answer)
    
    return array_of_answers


def questionFeatures(item):
    category = 'cat_' + item['category']            # shared feature - category
    sentence_pos = 'sent_' + item['sentence_pos']   # sentence position
    words = item['text'].split()                    # question text divided into words
    
    feats = Counter()
    for a in range(len(words)):
        feats['sc_' + words[a]] +=1
    feats[category] = 1
    feats[sentence_pos] = 1
    
    return feats


def generateVWData(data, outputFilename=None):
    h = open(outputFilename, 'w')
    for item in data:
        q_feats = questionFeatures(item)
        a_feats = answerFeatures(item)
        example = (q_feats, a_feats)
        writeVWExample(h, example, {})
    h.close()


# write a vw-style example to file
def writeVWExample(h, example, featureSetTracker=None):
    def sanitizeFeature(f):
        return re.sub(':', '_COLON_',
                      re.sub('\|', '_PIPE_',
                             re.sub('[\s]', '_', f)))

    def printFeatureSet(namespace, fdict):
        h.write(' |')
        h.write(namespace)
        for f, v in fdict.iteritems():
            h.write(' ')
            if abs(v) > 1e-6:
                ff = sanitizeFeature(f)
                h.write(ff)
                if abs(v - 1) > 1e-6:
                    h.write(':')
                    h.write(str(v))
                if featureSetTracker is None:
                    if not featureSetTracker.has_key(namespace): featureSetTracker[namespace] = {}
                    featureSetTracker[namespace][ff] = f

    (src, trans) = example
    if len(src) > 0:
        h.write('shared')
        printFeatureSet('s', src)
        h.write('\n')
    for i in range(len(trans)):
        (cost, tgt, pair) = trans[i]
        h.write(str(i + 1))
        h.write(':')
        h.write(str(cost))
        printFeatureSet('t', tgt)
        printFeatureSet('p', pair)
        h.write('\n')
    h.write('\n')


def trainVW(dataFilename, modelFilename, quietVW=False):
    cmd = vw_bin + ' -k -c -b 25 --holdout_off --passes 10 -q st --power_t 0.5 --csoaa_ldf m -d ' + dataFilename + ' -f ' + modelFilename
    if quietVW: cmd += ' --quiet'
    print 'executing: ', cmd
    p = os.system(cmd)
    if p != 0:
        raise Exception('execution of vw failed!  return value=' + str(p))


def testVW(dataFilename, modelFilename, quietVW=False):
    cmd = vw_bin + ' -t -q st -d ' + dataFilename + ' -i ' + modelFilename + ' -r ' + dataFilename + '.rawpredictions'
    if quietVW: cmd += ' --quiet'
    print 'executing: ', cmd
    p = os.system(cmd)
    if p != 0:
        raise Exception('execution of vw failed!  return value=' + str(p))

    h = open(dataFilename + '.rawpredictions')
    predictions = []

    this = []
    thisBestId = -1
    thisBestVal = 0
    for l in h.readlines():
        l = l.strip()
        res = l.split(':')
        if len(l) == 0:
            predictions.append((thisBestId - 1, predictions))
            this = []
            thisBestId = -1
            thisBestVal = 0
        elif len(res) == 2:
            class_id = int(res[0])
            class_val = float(res[1])
            if thisBestId < 0 or class_val < thisBestVal:
                thisBestId = class_id
                thisBestVal = class_val
            this.append((class_id, class_val))
        else:
            raise Exception('error on vw output, got line "' + l + '"')
    h.close()

    return predictions


if __name__ == "__main__":
    main()
