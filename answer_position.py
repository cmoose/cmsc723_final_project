#!/bin/python
import csv
import random
import final

def main():
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

if __name__ == '__main__':
    main()
