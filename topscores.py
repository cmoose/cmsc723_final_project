#!/bin/python
import csv
import random

def main():
    csvreader = csv.reader(open('data/test.csv'))
    csv_random_writer = csv.writer(open('data/submit_random.csv', 'wb'))
    csv_wiki_writer = csv.writer(open('data/submit_wiki.csv', 'wb'))
    csv_qanta_writer = csv.writer(open('data/submit_qanta.csv', 'wb'))
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
        csv_random_writer.writerow([q_id, ans[rindex]])
        csv_wiki_writer.writerow([q_id, wiki])
        csv_qanta_writer.writerow([q_id, qanta])

if __name__ == '__main__':
    main()
