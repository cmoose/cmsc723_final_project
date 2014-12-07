import csv
import numpy
import pickle
import os
from util import *
import re
from enum import Enum
import nltk

PKL_TRAIN = 'data/train.pkl'
PKL_DEV = 'data/dev.pkl'
PKL_STOPWORDS = 'data/stopwords.pkl'
TRAIN = 'data/train.csv'
TEST = 'data/test.csv'
VW_INPUT_TRAIN = 'vw/qa_vw.tr'
VW_INPUT_DEV = 'vw/qa_vw.de'
VW_MODEL = 'vw/qa_vw.model'
vw_bin = '/usr/local/bin/vw'
RAW_PRED = 'vw/qa_vw.de.rawpredictions'
SUBMISSION = 'data/submission.csv'
DEV_GUESSES = []
DEV_ANSWERS = []
ANSWER_LIST = []
ANSWER_MAP = {}
QUESTION_LIST = []
STOPWORDS = 'Stopwords.txt'
_question_mappings = {} #Keeps track of vw rowid -> [question ID,actual answer]
_qa_mappings = {}

class Data(Enum):
    train = 0
    dev = 1
    test = 2


def main(regenerate=False, submission=False):
    # answer_map = [] #Index is the map
    if not submission:
        if (not os.path.isfile(PKL_TRAIN)) or (not os.path.isfile(PKL_DEV) or regenerate):
            train = load_training_data(TRAIN)
            generate_train_dev(train)
        train = pickle.load(open(PKL_TRAIN, "rb"))
        dev = pickle.load(open(PKL_DEV, "rb"))
    else:
        train = full_data(load_training_data(TRAIN), True)
        dev = load_testing_data(TEST)

    create_answer_map(train)

    print 'Generating Classification Data'
    # training type_data
    generate_vw_data(train, Data.train, VW_INPUT_TRAIN)

    # testing type_data
    if not submission:
        generate_vw_data(dev, Data.dev, VW_INPUT_DEV)
    else:
        generate_vw_data(dev, Data.test, VW_INPUT_DEV)

    print 'Training Data'
    train_vw(VW_INPUT_TRAIN, VW_MODEL, True)

    print 'Outputting Test Results'
    test_vw(VW_INPUT_DEV, VW_MODEL, True)

    if not submission:
        predictions = compile_predictions()
        calc_score(predictions)
        exit()
        dev_results(len(dev))
    else:
        output_submission()


def calc_score(pred):
    total_qs = len(pred)
    num_correct = 0
    for p in pred:
        if p['answer'] == ['pred_ans']:
            num_correct+=1
    print num_correct
    print 'accy: ' + str(num_correct / float(total_qs) * 100) + '%'


def compile_predictions():
    with open(RAW_PRED, 'r') as training_guesses:
        raw_pred = training_guesses.readlines()
    q_row = 0
    q_stack = []
    predictions = [] 
    for line in raw_pred:
        if line != '\n':
            q_stack.append(line)
        else:
            #Reached a break
            best_raw_score = -100
            best_row = -1
            while len(q_stack)>0:
                #process question
                raw_ans = q_stack.pop(0)
                a_row = int(raw_ans.split(':')[0].strip())
                a_score = float(raw_ans.split(':')[1].strip())
                if a_score > best_raw_score:
                    best_raw_score = a_score
                    best_row = a_row - 1 #I think this is right, 
            q_info = _question_mappings[q_row]
            pred_ans = _qa_mappings[q_row][best_row]
            pred = {}
            pred['id'] = q_info[0]
            pred['answer'] = q_info[1]
            pred['pred_ans'] = pred_ans
            predictions.append(pred)
            q_row+=1
    return predictions

#input: num of questions in test set
def dev_results(len_dev):
    matrix = create_matrix()

    with open(RAW_PRED, 'r') as training_guesses:
        raw_pred = training_guesses.readlines()

    count = 0
    answer_count = 0
    max_answer = ''
    max_count = 0
    guess_set = []
    guess_scores = []
    for pred in raw_pred:
        if pred != '\n':
            answer_data = pred.strip().split(':')
            guess = DEV_GUESSES[count]
            answer = DEV_ANSWERS[answer_count]
            guess_set.append(guess)
            guess_scores.append(float(answer_data[1]))

            # # get the best guess
            if float(answer_data[1]) > max_count:
                max_count = float(answer_data[1])
                max_answer = guess

            if answer_data[0] == '1' and count != 0:
                x = ANSWER_MAP.get(max_answer)
                y = ANSWER_MAP.get(answer)

                if x and y:
                    matrix.itemset((x, y), matrix[x, y] + 1)
                else:
                    print answer, ' failed.'

                answer_count += 1
                max_count = 0
                max_answer = ''
                guess_set = []
                guess_scores = []
            count += 1

    x = ANSWER_MAP.get(max_answer) #guess
    y = ANSWER_MAP.get(answer) #answer

    if x and y:
        matrix.itemset((x, y), matrix[x, y] + 1)
    else:
        print answer, ' failed.'

    num_correct = numpy.trace(matrix)
    num_wrong = numpy.sum(numpy.sum(matrix, axis=0)) - num_correct
    # print numpy.sum(numpy.sum(matrix, axis=0))
    #print numpy.sum(numpy.sum(matrix, axis=1))
    numpy.set_printoptions(precision=2, suppress=True, linewidth=120)
    print 'accy: ' + str(num_correct / float(len_dev) * 100) + '%'


def output_submission():
    with open(RAW_PRED, 'r') as training_guesses:
        raw_pred = training_guesses.readlines()

    count = 0
    answer_count = 0
    max_answer = ''
    max_count = 0
    guess_set = []
    guess_scores = []
    for pred in raw_pred:
        if pred != '\n':
            answer_data = pred.strip().split(':')
            guess = DEV_GUESSES[count]
            guess_set.append(guess)
            guess_scores.append(float(answer_data[1]))

            # # get the best guess
            if float(answer_data[1]) > max_count:
                max_count = float(answer_data[1])
                max_answer = guess

            if count == 0:
                with open(SUBMISSION, 'w') as answers:
                    answers.write('Question ID,Answer\n')

            if answer_data[0] == '1' and count != 0:
                with open(SUBMISSION, 'a') as answers:
                    answers.write(QUESTION_LIST[answer_count] + ',' + max_answer + '\n')

                answer_count += 1
                max_count = 0
                max_answer = ''
                guess_set = []
                guess_scores = []
            count += 1

    with open(SUBMISSION, 'a') as answers:
        answers.write(QUESTION_LIST[answer_count] + ',' + max_answer + '\n')


def f_score(truth, prediction):
    tp = 0
    fp = 0
    fn = 0
    for k in range(1, len(truth)):
        if truth[k] == 1 and prediction[k] == 1:
            tp += 1
        if truth[k] == -1 and prediction[k] == -1:
            fp += 1
        if truth[k] == -1 and prediction[k] == 1:
            fn += 1
    prec = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    print('Precision')
    print(prec)
    print('Recall')
    print(recall)
    print('FSCORE')
    print((2 * prec * recall) / (prec + recall))
    return (2 * prec * recall) / (prec + recall)


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


def load_testing_data(filename):
    data = []
    with open(filename, 'rb') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in content:
            question_id = row[0]
            if question_id not in 'Question ID':
                text = row[1]
                quanta = format_scores(row[2].split(','))
                sentence_pos = row[3]
                wiki_score = format_scores(row[4].split(','))
                category = row[5]
                new_line = {'id': question_id, 'text': text, 'quanta': quanta,
                            'sentence_pos': sentence_pos,
                            'wiki': wiki_score, 'category': category}
                data.append(new_line)
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
    for item in data:
        for k, v in item['wiki'].items():
            if ANSWER_LIST.count(v) == 0:
                ANSWER_LIST.append(v)
        for k, v in item['quanta'].items():
            if ANSWER_LIST.count(v) == 0:
                ANSWER_LIST.append(v)
    for i in range(0, len(ANSWER_LIST)):
        ANSWER_MAP[ANSWER_LIST[i]] = i


def create_matrix():
    zeros = [[0 for x in range(0, len(ANSWER_LIST))] for x in range(0, len(ANSWER_LIST))]
    return numpy.matrix(zeros)


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


def full_data(data, mangle=False):
    selected_data = []
    for grouped_question in data.values():
        for question in grouped_question:
            selected_data.append(question)
    if mangle:
        numpy.random.shuffle(selected_data)
    return selected_data


# # type train = 1
# # type test = 0
def answer_features(item, data_type):
    array_of_answers = []
    get_labels(array_of_answers, item, 'wiki', data_type)
    get_labels(array_of_answers, item, 'quanta', data_type)

    if data_type == Data.dev:
        DEV_ANSWERS.append(item['answer'])
    return array_of_answers


def get_best_label(formatted_answers, item, label, data_type, normalize=True):
    import operator
    feats = Counter()
    max_value = max(item[label].iteritems(), key=operator.itemgetter(0))[0]
    feats['a_' + item[label][max_value]] = 1

    if normalize == True:
        wiki_max_prob = 141.312125
        quanta_max_prob = 0.933823174089
        if label == 'wiki':
            feats[label + '_prob'] = max_value / wiki_max_prob
        else:
            feats[label + '_prob'] = max_value / quanta_max_prob
    else:
        feats[label + '_prob'] = max_value

    new_answer = (1, feats, {})
    formatted_answers.append(new_answer)
    DEV_GUESSES.append(item[label][max_value])
    return formatted_answers


def get_labels(formatted_answers, item, label, data_type):
    for k, v in item[label].items():
        feats = Counter()
        if data_type == Data.dev or data_type == Data.test:
            DEV_GUESSES.append(v)
        feats['a_' + v] = 1
        feats[label + '_prob'] = k

        is_correct = 1  # False
        if data_type == Data.train:
            if v == item['answer']:
                is_correct = 0

        # append new answer to array_of_answers
        new_answer = (is_correct, feats, {})
        formatted_answers.append(new_answer)
    return formatted_answers


def build_stopwords():
    stopwords = []
    if not os.path.isfile(PKL_STOPWORDS):
        fh = open(STOPWORDS)
        for line in fh.readlines():
            stopwords.append(line.strip())
        pickle.dump(stopwords, open(PKL_STOPWORDS, 'wb'))
    else:
        stopwords = pickle.load(open(PKL_STOPWORDS))
    return stopwords


def question_features(item):
    feats = Counter()
    category = 'cat_' + item['category']  # shared feature - category
    sentence_pos = 'sent_' + item['sentence_pos']  # sentence position
    q_id = 'id_' + item['id']
    feats[q_id] = 1
    feats[category] = 1
    feats[sentence_pos] = 1

    stopwords = build_stopwords()
    sentences = nltk.sent_tokenize(item['text'])
    for sentence in sentences:
        raw_tokens = nltk.word_tokenize(sentence)

        # POS
        #tagged_tokens = nltk.pos_tag(raw_tokens)
        #for a in range(len(tagged_tokens)):
        #    feats['sc_' + tagged_tokens[a][0] + '_' + tagged_tokens[a][1]] += 1

        #Stopwords
        #tokens = []
        #for token in raw_tokens:
        #    if stopwords.count(token.strip()) == 0:
        #        tokens.append(token.strip())

        # Bag of words
        for a in range(len(raw_tokens)):
            feats['sc_' + raw_tokens[a]] += 1

        # n_gram
        #for n in range(2, 4):
        #    n_gram = nltk.ngrams(tokens, n)
        #    for gram in n_gram:
        #        feats['n%s_%s' % (str(n), repr(gram[0]))] += 1

    return feats


# # train type = 1
# # test type = 0
def generate_vw_data(data, data_type, output_filename=None):
    with open(output_filename, 'w') as h:
        q_vw_rowid = 0 #rowid of question in vw, needed to match up output
        for item in data:
            if data_type == Data.dev or data_type == Data.test:
                _qa_mappings[q_vw_rowid] = {}
                _question_mappings[q_vw_rowid] = [item['id'], item['answer']]
            q_feats = question_features(item)
            a_feats = answer_features(item, data_type)
            example = (q_feats, a_feats)
            write_vw_example(h, example, data_type, q_vw_rowid, {})
            if data_type == Data.dev or data_type == Data.test:
                QUESTION_LIST.append(item['id'])
            q_vw_rowid+=1


# write a vw-style example to file
def write_vw_example(h, example, data_type, q_vw_rowid, feature_set_tracker=None):
    def get_article(tgt):
        for f,v in tgt.items():
            if re.search("^a_", f):
                return f[2:]
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
        
        #Hack to get vw row information we need
        if data_type != Data.train:
            article = get_article(tgt)
            _qa_mappings[q_vw_rowid][i] = article
        
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
