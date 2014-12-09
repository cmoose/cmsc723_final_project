#!/bin/python
import csv
import random
import final

def main():
    data = final.full_data(final.load_training_data('data/train.csv'), True)
    fhw = open('data/ans_positions.csv', 'wb')
    for question in data:
        q_id = question['id']
        ans = question['answer']
        i = 1
        wiki_pos = -1
        for k,v in question['wiki'].items():
            if ans == v:
                wiki_pos = i
                break
            i+=1
        j = 1
        quanta_pos = -1
        for k,v in question['quanta'].items():
            if ans == v:
                quanta_pos = j
            j+=1
        fhw.write('%s,%d,%d\n' % (q_id,wiki_pos,quanta_pos))

if __name__ == '__main__':
    main()
