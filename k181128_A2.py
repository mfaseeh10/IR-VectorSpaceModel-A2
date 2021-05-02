from tkinter import ttk
from tkinter import messagebox
import tkinter
from collections import Counter
import nltk
import string
import numpy as np
import math

alpha = 0.005


# This function will do casefolding .
def case_folding(w):
    return (w.lower())

# This function will remove punctuations from  words


def remove_punctuation(w):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~‘’“”'''
    no_punct = ""
    for char in w:
        if char not in punctuations:
           # print(char)
            no_punct = no_punct + char
    w = no_punct

    return (w)

# This function will stemming using PorterStemmer algorithm using nltk library


def stemming(w):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    return (ps.stem(w))

# A caller function to invoke all preprocessing functions

def pre_processing(qu):

    wList = make_word_list(qu)
    pre_propList = []
    for w in wList:
        w = case_folding(w)
        w = remove_punctuation(w)
        stopwordsList = read_stopwords()
        if w not in stopwordsList:
            w = stemming(w)
            pre_propList.append(w)
          
   
    return (pre_propList)

# This function will return list of all stop words .


def read_stopwords():
    stopword = []
  #  f = open("D:/faseeh/SEMESTER 6/IR/A1/Stopwords/Stopword-List.txt", "r")
    f = open(
        '/home/mfaseeh10/Documents/Python /IR_A2k181128/Stopwords/Stopword-List.txt', 'r')
    stopwordlist = f.readlines()
    f.close()
    for l in stopwordlist:
        stopword = stopword + l.split()
    return (stopword)


# function to break string into list of words
def make_word_list(word_list):
    W = []
    w = ''
    for word in word_list:
        if ((word != ' ') and (word != '.') and (word != ']') and (word != '\n') and (word != '-') and (
                word != '—') and (word != '?') and (word != '"') and (word != '…') and (word != '/')):
            w = w + word
        elif ((w != '')):
            W = W + [w]
            w = ''
    if ((w != '')):
        W = W + [w]
    return (W)

# Function to return an id for a given document


def get_doc_id(fn):
    if (fn[1] == '.'):
        return (fn[0])
    else:
        return (fn[0] + fn[1])


# function to make a dictionary with doc id and its title
def make_doc_titles_dict():
    doc_titles = []
    for i in range(1, 51):
        rfile = open(
            '/home/mfaseeh10/Documents/Python /IR_A2k181128/ShortStories/' + str(i) + '.txt', 'r')
        doc = rfile.readlines()
        if(i == 50):
            doc_titles.append(doc[0][:-1]+' '+doc[1][:-1]) 
        else:
            doc_titles.append(doc[0][:-1]) 

        # doc_titles.append(doc[0])
        # print(doc[0])
    # print(doc_titles)

    # for i in range(1,51):
    #    print(doc_titles[i])

    return doc_titles

# preprocessed titles of each document
def make_processed_titles(st):
    p_titles = []
    stopwordsList = read_stopwords()
    
    #print(st) 
    wordlist = []  # to store file as a list of words
    for t in st:
        #print(t)
        wList = make_word_list(t)
        #print(wList)
        wordlist.append(wList)

    #print(wordlist)
    for i in range(len(wordlist)):
        cleaned_word = []  # to store list of pre processed words
           
        for word in wordlist[i]:

            if (word == ''):
                continue
            
            word = case_folding(word)
            word = remove_punctuation(word)

            #checked again to see if empty string is present after punctuation removal
            if (word == ''):
                continue

            if word not in stopwordsList:
                word = stemming(word)
                cleaned_word.append(word)
        #print(cleaned_word)
        p_titles.append(cleaned_word)

    
    
    return p_titles



def make_processed_dataset(processed_data):
    import os
    #global processed_data
    directory = '/home/mfaseeh10/Documents/Python /IR_A2k181128/ShortStories/'
    # print(directory)
    dirList = os.listdir(directory)
    dirList.sort()
    #print(dirList)
    fileIDs = []
    for fn in dirList:
        fileIDs.append(int(get_doc_id(fn)))

    fileIDs.sort()
    
    #print(fileIDs)   
    
    stopwordsList = read_stopwords()

    #for filename in dirList:
    for filename in fileIDs:    
        #if (filename.endswith(".txt")):
            
            lines = []  # to store each line in file as a list
            f = open('./ShortStories/' + str(filename) + '.txt', 'r', encoding="utf8")
            lines = f.readlines()
            f.close()

            wordlist = []  # to store file as a list of words
            for line in lines:
                wList = make_word_list(line)
                wordlist.extend(wList)

            cleaned_word = []  # to store list of pre processed words
            for word in wordlist:
                if (word == ''):
                    continue

            #    word = pre_processing(word)

                word = case_folding(word)
                word = remove_punctuation(word)

                if word not in stopwordsList:
                    word = stemming(word)
                    cleaned_word.append(word)

            processed_data.append(cleaned_word)
            # print(cleaned_word)

            #processed_data.sort()

    return processed_data

#function to return doc freq of a given token
def doc_freq(DF, word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


#extracting data as title and indexing
story_titles = make_doc_titles_dict()
processed_titles = make_processed_titles(story_titles)

processed_data1 = []

#TO PROCESSS STORIES DATA
stories = make_processed_dataset(processed_data1)


DF = {}  #to store number of doc ids of each word
for i in range(len(stories)):
    tokens = stories[i]

    for w in tokens:
        try:
            DF[w].add(i+1)     
        except:
            DF[w] = {i+1}


#to write dictionary indexes into file
# try:
#     ii_file = open(
#     '/home/mfaseeh10/Documents/Python /IR_A2k181128/Indexes/dictionary.txt', 'wt')
#     ii_file.write(str(DF))
#     ii_file.close()
# except:
#     print("Unable to write to file")

#maintains doc freq of each ters
DFcount = {}
for i in DF:
    DFcount[i] = len(DF[i])

#to calculate tf_idf of body
N = len(stories) #total no of Docs 
doc = 1
tf_idf = {}
for i in range(len(stories)):
    
    tokens = stories[i]
    #print(story_titles[i+1])
    #break
    
    counter = Counter(tokens)
    #print('counter: ', counter)
    words_count = len(tokens)
    #print('words_count: ', words_count)

    
    for token in np.unique(tokens):
        #print(token)
        tf = counter[token]/words_count
        df = doc_freq(DFcount, token)
        idf = np.log(N/(df+1))
    
        tf_idf[doc, token] = tf*idf
        
        
    doc += 1


#to calculate tf_idf of titles
doc = 1
tf_idf_title = {}

for i in range(len(processed_titles)):
    
    tokens = processed_titles[i]
    #print(tokens)
    counter = Counter(tokens + stories[i])
    words_count = len(tokens + stories[i])

    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(DFcount, token)
        idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
        
        tf_idf_title[doc, token] = tf*idf

    doc += 1


#no of words in the dataset
total_vocabSize = len(DF)
print(total_vocabSize)

#total words list in the dataset
total_vocab = [x for x in DF]

#multiply tf_idf by alpha
for i in tf_idf:
    tf_idf[i] *= alpha

#give value of tf_idf title to tf_idf is word exist in both title and body
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

#matching score method
def matching_score(k, query):
    tokens = pre_processing(query)

    print("Matching Score")
    print("\nQuery:", query)
    print("")
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")
    
    l = []
    
    for i in query_weights[:10]:
        l.append(i[0])
    
    print(l)
    


#to computer cosine similartity between doc and query
def cosine_sim(a, b):
    
    #store product of magnitudes
    mag = (abs(np.linalg.norm(a))*abs(np.linalg.norm(b)))
    
    if(mag < 0 or mag == 0):
        return 0

    cos_sim = np.dot(a, b)/mag
    return cos_sim


#a datastructure to store vectors of each document in vector 
D = np.zeros((len(stories), total_vocabSize))

for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass


#function to generate vector of the query
def gen_vector(tokens):
    
    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(DFcount,token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q


def cosine_similarity(k, query):
    print("Cosine Similarity")
    tokens = pre_processing(query)
 
    print("\nQuery:", query)
    print("")
    print(tokens)
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    print(query_vector)
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))

    
    tem_out = []
    for el in d_cosines:
        if(el <= 0):
            continue
        #if(el <= 0.005):
        tem_out.append(d_cosines.index(el))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    print("")
    print(tem_out)

    
    return tem_out

#Q = cosine_similarity(49, "across adventures")



#================================================== UI ==================================================================
#================================================== for ==================================================================
#================================================== PROGRAM ==================================================================

import tkinter
from tkinter import ttk

window=tkinter.Tk()
window.title("Vector Space Model")

labelone=ttk.Label(window,text="Enter Query : ")
labelone.grid(row = 0, column = 0)

labeltwo=ttk.Label(window,text="Result of Query by Vector Space Model is : ")
labeltwo.grid(row = 2, column = 0)


inp=tkinter.StringVar()


userentry=ttk.Entry(window,width=50,textvariable = inp)
userentry.grid(row = 0 , column = 1)


def action():

    Q = cosine_similarity(49,inp.get())
    
    s='Length = '+str(len(Q))        
    labelthree=ttk.Label(window,text=s)
    labelthree.grid(row = 2, column = 1)
    
    
    if(len(Q)==0):
        labelfive=ttk.Label(window,text='No Result Found .')
    else:
        labelfive=ttk.Label(window,text=Q)
    labelfive.grid(row = 3, column = 1)
    
    
    labelthree.after(10000,lambda: labelthree.destroy())
    labelfive.after(10000,lambda: labelfive.destroy())


btn = ttk.Button(window,text="Submit",command=action)
btn.grid(row = 0,column = 2)



window.mainloop()
 