#!/bin/python
import csv
import random
import final

def create_minimal_csv(filename):
    csvreader = csv.reader(open(filename))
    csvwriter = csv.writer(open(filename + '.new', 'wb'))
    new_line = []
    header = ['Question ID','Question Text','QANTA Scores','Answer','Sentence Position','IR_Wiki Scores','category']
    for line in csvreader:
        q_id = line[0]
        q_text = line[1]
        q_score = line[2].split(',')[0]
        q_ans = line[3]
        q_pos = line[4]
        q_wscore = line[5].split(',')[0]
        q_cat = line[6]
        csvwriter.writerow([q_id, q_text, q_score, q_ans, q_pos, q_wscore, q_cat])


def create_minimal_test_csv(filename):
    csvreader = csv.reader(open(filename))
    csvwriter = csv.writer(open(filename + '.new', 'wb'))
    new_line = []
    header = ['Question ID','Question Text','QANTA Scores','Sentence Position','IR_Wiki Scores','category']
    for line in csvreader:
        q_id = line[0]
        q_text = line[1]
        q_score = line[2].split(',')[0]
        q_pos = line[3]
        q_wscore = line[4].split(',')[0]
        q_cat = line[5]
        csvwriter.writerow([q_id, q_text, q_score, q_pos, q_wscore, q_cat])


def find_ans_positions():
    csvreader = csv.reader(open('data/train.csv'))
    fhw = open('data/ans_positions.csv', 'wb')
    data = []
    for line in csvreader:
        q_id = line[0]
        quanta_answers_str = line[2]
        quanta_answers = []
        correct_answer = line[3]
        wiki_answers_str = line[5]
        wiki_answers = []
        for qa in quanta_answers_str.split(','):
            quanta = qa.split(':')[0]
            quanta_answers.append(quanta)
        for wa in wiki_answers_str.split(','):
            wiki = wa.split(':')[0]
            wiki_answers.append(wiki)
        data.append([q_id,correct_answer,quanta_answers,wiki_answers])
    for question in data:
        q_id = question[0]
        ans = question[1]
        quanta_answers = question[2]
        wiki_answers = question[3]
        i = 1
        wiki_pos = -1
        wiki_array = []
        for wa in wiki_answers:
            if ans == wa:
                wiki_pos = i
                break
            i+=1
        j = 1
        quanta_pos = -1
        quanta_array = []
        for qa in quanta_answers:
            if ans == qa:
                quanta_pos = j
            j+=1
        fhw.write('%s,%d,%d\n' % (q_id,wiki_pos,quanta_pos))

def main():
    create_minimal_csv('data/train.csv')
    create_minimal_test_csv('data/test.csv')

if __name__ == '__main__':
    main()
