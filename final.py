import csv
import numpy
import pickle
import os
from sys import maxint
from util import *
import re
from enum import Enum
import nltk
import wikipedia

PKL_TRAIN = 'data/train.pkl'
PKL_DEV = 'data/dev.pkl'
PKL_STOPWORDS = 'data/stopwords.pkl'
PKL_Q_NOUNS = 'data/qnouns.pkl' #Keep a list of nouns per question for perf
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
ANSWER_MAP = {}
QUESTION_LIST = []
STOPWORDS = 'Stopwords.txt'
WP_DOCS = {}
WP_COMMA_ARTICLES = []
Q_NOUNS = {}


class Data(Enum):
    train = 0
    dev = 1
    test = 2


def main(regenerate=False, testing=True):
    #load wikipedia data
    #load_wikipedia()
    #build_q_nouns()

    if not testing:
        if (not os.path.isfile(PKL_TRAIN)) or (not os.path.isfile(PKL_DEV) or regenerate):
            train = load_training_data(TRAIN)
            generate_train_dev(train)

        train = pickle.load(open(PKL_TRAIN, "rb"))
        dev = pickle.load(open(PKL_DEV, "rb"))
    else:
        train = full_data(load_training_data(TRAIN), True)
        dev = load_testing_data(TEST)

    print 'Generating Classification Data'
    # training type_data
    generate_vw_data(train, Data.train, VW_INPUT_TRAIN)

    print 'Generating dev/test data'
    # testing type_data
    if not testing:
        generate_vw_data(dev, Data.dev, VW_INPUT_DEV)
    else:
        generate_vw_data(dev, Data.test, VW_INPUT_DEV)

    print 'Training Data'
    train_vw(VW_INPUT_TRAIN, VW_MODEL, True)

    print 'Outputting Test Results'
    test_vw(VW_INPUT_DEV, VW_MODEL, True)

    if not testing:
        dev_results(len(dev), Data.dev)
    else:
        dev_results(len(dev), Data.test)


def dev_results(num_questions, data_type):
    with open(RAW_PRED, 'r') as training_guesses:
        prediction_scores = training_guesses.readlines()

    # separate predictions into list entries.
    count = 0
    predictions_by_question = [[] for x in range(0, num_questions)]
    for prediction_score in prediction_scores:
        if prediction_score != '\n':
            predictions_by_question[count].append({'prediction_score': float(prediction_score.split(':')[1].rstrip())})
        else:
            count += 1

    # add labels to guesses
    count = 0
    for question in predictions_by_question:
        for prediction_score in question:
            prediction_score['label'] = DEV_GUESSES[count]
            count += 1

    # retrieve max solution
    vpw_guess = {}
    count = 0
    for question in predictions_by_question:
        max_value = maxint
        max_object = {}
        for prediction in question:
            if prediction['prediction_score'] < max_value:
                max_value = prediction['prediction_score']
                max_object = prediction
            vpw_guess[count] = max_object
        count += 1

    if data_type == Data.dev:
        # compute accy
        correct = 0
        wrong = 0
        for i in range(0, num_questions):
            prediction = vpw_guess[i]['label']
            if prediction == DEV_ANSWERS[i]:
                correct += 1
            else:
                wrong += 1

        print (correct / float(num_questions))*100, '% accuracy'

    if data_type == Data.test:
        with open(SUBMISSION, 'w') as answers:
            answers.write('Question ID,Answer\n')
        for i in range(0, num_questions):
            with open(SUBMISSION, 'a') as answers:
                prediction = vpw_guess[i]['label']
                answers.write(QUESTION_LIST[i] + ',' + prediction + '\n')


def build_q_nouns():
    if (not os.path.isfile(PKL_Q_NOUNS)):
        print "Building questions' nouns for all data"
        pickfh = open(PKL_Q_NOUNS, 'wb')
        questions = {} #key: q_id, values: nouns in question
        train = full_data(load_training_data(TRAIN), True)
        dev = load_testing_data(TEST)
        i = 1
        for item in train:
            print "Getting nouns for %d/%d" % (i,len(train))
            nouns = get_nouns(item['text'])
            questions[item['id']] = nouns
            i+=1
        j = 1
        for item in dev:
            print "Getting nouns for %d/%d" % (j,len(dev))
            nouns = get_nouns(item['text'])
            questions[item['id']] = nouns
            j+=1
        pickle.dump(questions, pickfh)
    load_q_nouns()


def load_q_nouns():
    global Q_NOUNS
    Q_NOUNS = pickle.load(open(PKL_Q_NOUNS))


def load_wikipedia():
    global WP_DOCS
    global WP_COMMA_ARTICLES
    WP_DOCS = wikipedia.load_pickle_files()
    WP_COMMA_ARTICLES = wikipedia.load_comma_pickle_file()


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
    if data_type == Data.test or data_type == Data.dev:
        get_labels(array_of_answers, item, 'wiki', data_type)
        get_labels(array_of_answers, item, 'quanta', data_type)
    else:
        get_labels(array_of_answers, item, 'wiki', data_type)
        get_labels(array_of_answers, item, 'quanta', data_type)

    if data_type == Data.dev:
        DEV_ANSWERS.append(item['answer'])
    return array_of_answers


#input question text, get back nouns
def get_nouns(q_text):
    tokens = nltk.word_tokenize(q_text)
    tagged_tokens = nltk.pos_tag(tokens)
    noun_tokens = []
    #Pull out all nouns from the question
    for item in tagged_tokens:
        if re.search("NN", item[1]):
            noun_tokens.append(item[0])
    return noun_tokens


def get_cached_nouns(q_id):
    global Q_NOUNS
    if len(Q_NOUNS) == 0:
        print 'Cache is missing, rebuilding...'
        build_q_nouns()
    return Q_NOUNS[q_id]


def lookup_article_title(answer):
    answer = answer.replace(')', '')
    answer = answer.replace('(', '')
    for article in WP_COMMA_ARTICLES:
        #print answer
        if re.search(answer, article):
            return article
    return None

def get_wp_word_count(q_nouns, answer):
    prefix = answer[0]
    count = 0
    for word in q_nouns:
        if WP_DOCS.has_key(prefix):
            if WP_DOCS[prefix].has_key(answer):
                if WP_DOCS[prefix][answer].has_key(word):
                    count+=1
            else:
                #missing article, return smoothing value count of 3
                return 3
        else:
            #Found a truncated article title, find the real title, and lookup that
            full_answer = lookup_article_title(answer)
            if full_answer:
                prefix = full_answer[0]
                if WP_DOCS[prefix][full_answer].has_key(word):
                    count+=1
    return count


def get_best_label(formatted_answers, item, label, data_type, normalize=True):
    import operator

    feats = Counter()
    max_value = max(item[label].iteritems(), key=operator.itemgetter(0))[0]
    feats['a_' + item[label][max_value]] = 1

    #wikipedia features
    #answer = item[label][max_value]
    #q_nouns = get_nouns(item['text'])
    #q_nouns = get_cached_nouns(item['id'])
    #noun_word_count = get_wp_word_count(q_nouns, answer)
    #feats['wp_q_word_count'] = noun_word_count

    if normalize == True:
        wiki_max_prob = 141.312125
        quanta_max_prob = 0.934937484
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


def get_labels(formatted_answers, item, label, data_type, normalize=False):
    #q_nouns = get_nouns(item['text'])
    #q_nouns = get_cached_nouns(item['id'])
    for k, v in item[label].items():
        feats = Counter()
        if data_type == Data.dev or data_type == Data.test:
            DEV_GUESSES.append(v)
        feats['a_' + v] = 1

        #wikipedia feature
        #noun_word_count = get_wp_word_count(q_nouns, v)
        #feats['wp_q_word_count'] = noun_word_count

        if normalize == True:
            wiki_max_prob = 305.988897
            quanta_max_prob = 0.934937484
            if label == 'wiki':
                feats[label + '_prob'] = k / wiki_max_prob
            else:
                feats[label + '_prob'] = k / quanta_max_prob
        else:
            feats[label + '_prob'] = k

        is_correct = 1  # False
        if data_type == Data.train and v == item['answer']:
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
    feats[category] = 1
    feats[sentence_pos] = 1

    stopwords = build_stopwords()
    sentences = nltk.sent_tokenize(item['text'])
    for sentence in sentences:
        raw_tokens = nltk.word_tokenize(sentence)

        #Stopwords
        #tokens = []
        #for token in raw_tokens:
        #    if stopwords.count(token.strip()) == 0:
        #        tokens.append(token.strip())

        # POS
        #tagged_tokens = nltk.pos_tag(raw_tokens)
        #for a in range(len(tagged_tokens)):
        #    if tagged_tokens[a][1] == 'NN':
        #        feats['scp_' + tagged_tokens[a][0] + '_' + tagged_tokens[a][1]] += 1

        # Bag of words
        #for a in range(len(tokens)):
        #    feats['sc_' + tokens[a]] += 1

        # n_gram
        #for n in range(2, 4):
        #    n_gram = nltk.ngrams(raw_tokens, n)
        #    for gram in n_gram:
        #        feats['n%s_%s' % (str(n), repr(gram[0]))] += 1

    return feats


# # train type = 1
# # test type = 0
def generate_vw_data(data, data_type, output_filename=None):
    with open(output_filename, 'w') as h:
        i = 1
        for item in data:
            #print 'New Question %d/%d' % (i, len(data))
            q_feats = question_features(item)
            a_feats = answer_features(item, data_type)
            example = (q_feats, a_feats)
            write_vw_example(h, example, {})
            if data_type == Data.dev or data_type == Data.test:
                QUESTION_LIST.append(item['id'])
            i+=1


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
