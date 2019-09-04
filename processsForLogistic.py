import nltk
import csv
from nltk.tokenize import word_tokenize

def generateCSV():
    with open("token.csv", 'w', newline='',encoding='utf-8') as f, open('processed_test1.txt', 'r', errors='ignore') as tweet:
        head=['post','tags']
        csv_write = csv.writer(f)
        csv_head = head
        csv_write.writerow(csv_head)
        for sentence in tweet.readlines():
            tok=[]
            l=word_tokenize(sentence)
            tag=l[0]
            l.pop(0)
            text=' '.join(l)
            tok.append(text)
            tok.append(tag)
            csv_write.writerow(tok)
    f.close()
    tweet.close()

def generateCSV_forTest():
    with open("token_test.csv", 'w', newline='',encoding='utf-8') as f,open('processed_test1.txt', 'r', encoding='utf-8', errors='ignore') as tt:
        csv_write = csv.writer(f)
        head=[]
        head.append('post')
        csv_write.writerow(head)
        for s in tt.readlines():
            tok = []
            l = word_tokenize(s)
            text = ' '.join(l)
            tok.append(text)
            # print(s)
            csv_write.writerow(tok)
    tt.close()

# generateCSV()
generateCSV_forTest()
