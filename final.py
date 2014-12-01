import csv
import numpy
import pickle
import os
from util import *

PKL_TRAIN = 'data/train.pkl'
PKL_DEV = 'data/dev.pkl'
TRAIN = 'data/train.csv'


def main():
    answer_map = [] #Index is the map
    if (not os.path.isfile(PKL_TRAIN)) | (not os.path.isfile(PKL_DEV)):
        train = load_training_data(TRAIN)
        generate_train_dev(train)

    train = pickle.load(open(PKL_TRAIN, "rb"))
    dev = pickle.load(open(PKL_DEV, "rb"))
    answer_map = create_answer_map(train)

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

def questionFeatures(item):
    category = item['category']
    feats = Counter()
    feats[category] = 1

    return feats

if __name__ == "__main__":
    main()
