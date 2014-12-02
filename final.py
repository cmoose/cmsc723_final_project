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
    # answer_map = [] #Index is the map
    if (not os.path.isfile(PKL_TRAIN)) | (not os.path.isfile(PKL_DEV)):
        train = load_training_data(TRAIN)
        generate_train_dev(train)

    train = pickle.load(open(PKL_TRAIN, "rb"))
    dev = pickle.load(open(PKL_DEV, "rb"))
    # answer_map = create_answer_map(train)

    print 'Generating Classification Data'
    # training type_data
    generate_vw_data(train, 1, VW_INPUT_TRAIN)
    # testing type_data
    generate_vw_data(dev, 0, VW_INPUT_DEV)

    print 'Training Data'
    train_vw(VW_INPUT_TRAIN, VW_MODEL, True)

    print 'Outputting Test Results'
    test_vw(VW_INPUT_DEV, VW_MODEL, True)

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
                            'wiki': wiki_score, 'category': category}
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
        for k, v in item['wiki'].items():
            if answer_map.count(v) == 0:
                answer_map.append(v)
        for k, v in item['quanta'].items():
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
        selected_data.append(item[len(item) - 1])
    numpy.random.shuffle(selected_data)
    split = int(len(selected_data) * .80)
    train = selected_data[:split]
    dev = selected_data[split + 1:]
    return train, dev


# # type train = 1
# # type test = 0
def answer_features(item, data_type):
    array_of_answers = []
    correct_answer = item['answer']
    get_labels(array_of_answers, item, 'wiki', correct_answer, data_type)
    get_labels(array_of_answers, item, 'quanta', correct_answer, data_type)
    return array_of_answers


def get_labels(formatted_answers, item, label, correct_answer, data_type):
    for k, v in item[label].items():
        feats = Counter()
        feats['a_' + v] = 1
        feats[label + '_prob'] = k

        is_correct = 1  # False
        if v == correct_answer and data_type:
            is_correct = 0

        # append new answer to array_of_answers
        new_answer = (is_correct, feats, {})
        formatted_answers.append(new_answer)
    return formatted_answers


def question_features(item):
    category = 'cat_' + item['category']  # shared feature - category
    sentence_pos = 'sent_' + item['sentence_pos']  # sentence position
    words = item['text'].split()  # question text divided into words

    feats = Counter()
    for a in range(len(words)):
        feats['sc_' + words[a]] += 1
    feats[category] = 1
    feats[sentence_pos] = 1

    return feats


# # train type = 1
## test type = 0
def generate_vw_data(data, data_type, output_filename=None):
    with open(output_filename, 'w') as h:
        for item in data:
            q_feats = question_features(item)
            a_feats = answer_features(item, data_type)
            example = (q_feats, a_feats)
            write_vw_example(h, example, {})


# write a vw-style example to file
def write_vw_example(h, example, feature_set_tracker=None):
    def sanitize_feature(f):
        return re.sub(':', '_COLON_',
                      re.sub('\|', '_PIPE_',
                             re.sub('[\s]', '_', f)))

    def print_feature_set(namespace, fdict):
        h.write(' |')
        h.write(namespace)
        for f, v in fdict.iteritems():
            h.write(' ')
            if abs(v) > 1e-6:
                ff = sanitize_feature(f)
                h.write(ff)
                if abs(v - 1) > 1e-6:
                    h.write(':')
                    h.write(str(v))
                if feature_set_tracker is None:
                    if not feature_set_tracker.has_key(namespace):
                        feature_set_tracker[namespace] = {}
                    feature_set_tracker[namespace][ff] = f

    (src, trans) = example
    if len(src) > 0:
        h.write('shared')
        print_feature_set('s', src)
        h.write('\n')
    for i in range(len(trans)):
        (cost, tgt, pair) = trans[i]
        h.write(str(i + 1))
        h.write(':')
        h.write(str(cost))
        print_feature_set('t', tgt)
        print_feature_set('p', pair)
        h.write('\n')
    h.write('\n')


def train_vw(data_filename, model_filename, quiet_vw=False):
    cmd = vw_bin + ' -k -c -b 25 --holdout_off --passes 10 -q st --power_t 0.5 --csoaa_ldf m -d ' + data_filename + ' -f ' + model_filename
    run_vw(cmd, quiet_vw)


def test_vw(data_filename, model_filename, quiet_vw=False):
    cmd = vw_bin + ' -t -q st -d ' + data_filename + ' -i ' + model_filename + ' -r ' + data_filename + '.rawpredictions'
    run_vw(cmd, quiet_vw)


def run_vw(cmd, quiet_vw):
    if quiet_vw:
        cmd += ' --quiet'
    print 'executing: ', cmd
    p = os.system(cmd)
    if p != 0:
        raise Exception('execution of vw failed!  return value=' + str(p))


if __name__ == "__main__":
    main()
