#!/bin/python
import csv
import random

def main():
    csvreader = csv.reader(open('data/test.csv'))
    csvwriter = csv.writer(open('data/submit.csv', 'wb'))
    for line in csvreader:
        #Question ID,Question Text,QANTA Scores,Sentence Position,IR_Wiki Scores,category
        q_id = line[0]
        ans = []
        qanta = line[2].split(',')[0].split(':')[0]
        ans.append(qanta)
        wiki = line[4].split(',')[0].split(':')[0]
        ans.append(wiki)
        #choose a random top score: qanta/wiki
        rindex = random.randint(0,1)
        #write to csv
        csvwriter.writerow([q_id, ans[rindex]])

if __name__ == '__main__':
    main()
