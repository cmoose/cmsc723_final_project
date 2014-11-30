import csv
import numpy
import pickle
import os

PKL_TRAIN = 'data/train.pkl'
PKL_DEV = 'data/dev.pkl'
TRAIN = 'data/train.csv'


def main():
    if (not os.path.isfile(PKL_TRAIN)) | (not os.path.isfile(PKL_DEV)):
        train = load_training_data(TRAIN)
        generate_train_dev(train)

    train = pickle.load(open(PKL_TRAIN, "rb"))
    dev = pickle.load(open(PKL_DEV, "rb"))

    print len(train)
    print len(dev)


def load_training_data(filename):
    data = []
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
                data.append(new_line)
    return data


def format_scores(data):
    formatted_data = {}
    for item in data:
        split_data = item.split(':')
        if len(split_data) == 2:
            answer = split_data[0]
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
    numpy.random.shuffle(data)
    split = int(len(data) * .80)
    train = data[:split]
    dev = data[split + 1:]
    return train, dev


if __name__ == "__main__":
    main()