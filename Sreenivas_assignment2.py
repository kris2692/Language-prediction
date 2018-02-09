# -*- coding: utf-8 -*-
#Name: Krishna Sreenivas
#Student ID: 800984436
"""
Created on Fri Sep 29 13:42:46 2017

@author: krish
"""

import nltk
import string
from nltk.corpus import udhr  
from collections import Counter
from nltk import FreqDist
import math

#function to preprocess text, lowers each characters and considers only alpha-numeric.
def text_processing(array):
#    array = array.translate(array.maketrans('', '', string.punctuation))
    array1=''.join(char.lower() for char in array if char.isalnum())
    return list(array1)

#calculates frequency of each character except space
def char_count(array1):
    fd= nltk.FreqDist(array1)
    if fd.keys()==' ':
        fd.pop(fd.keys())
    return fd
            
#constructs a dictionary of unigram probabilities.
def unigram_count(dictionary):
    unigram_count_dict={}
    for word in dictionary:
        if word!=' ':
            unigram_count_dict[word]=math.log(dictionary[word]/sum(dictionary.values()))
    return unigram_count_dict
                
#constructs a dictionary of bigram probabilities.
def bigram_count(array,dictionary):
    bigram_count_dict={}
    fd=nltk.FreqDist(array)
    for word1,word2 in fd.keys():
        if word1!=' ' and word2!=' ':
            for word in dictionary.keys():
                if word==word1:
                    bigram_count_dict[word1,word2]=math.log((fd[word1,word2]/dictionary.get(word)))
    return bigram_count_dict,fd
                    
#constructs a dictionary of trigram probabilities.
def trigram_count(array,dictionary):
    trigram_count_dict={}
    fd=nltk.FreqDist(array)
    for word1,word2,word3 in fd.keys():
        if word1!=' ' and word2!=' ' and word3!=' ':
            for word4,word5 in dictionary.keys():
                if word1==word4 and word2==word5: 
                    trigram_count_dict[word1,word2,word3]=math.log((fd[word1,word2,word3])/(dictionary[word4,word5]))
    return trigram_count_dict

#unigram prediction model
def lang1_vs_lang2_unigram(lang1,lang_dict1,lang_dict2):
    temp=''
    temp1=[]#holds probability value of a word based on language1 unigram probabilities.
    temp2=[]#holds probability value of a word based on language2 unigram probabilities.
    a=0
    b=0
    unigram_lang1_accuracy = 0#counters to hold accuracy values
    unigram_lang2_accuracy = 0
    for word in lang1: #for each word in test case
        if word.isalnum(): #not considering punctuations 
            temp=word.lower() #lowering each char         
            for char1 in temp:
                for char2 in lang_dict1.keys():
                    if char1!=char2: #appending 0 if the characters in test and language1 dictionary are not matching
                        temp1.append(0)
                    if char1==char2: #if char in test and language 1 dictionary matches 
                        temp1.append(lang_dict1[char1])
                for char3 in lang_dict2.keys():
                    if char1!=char3: #appending 0 if the characters in test and language2 dictionary are not matching
                        temp2.append(0)
                    if char1==char3:#if char in test and language 2 dictionary matches
                        temp2.append(lang_dict2[char1])
            for value2 in temp2:
                b+=value2 #adding up the logarithmic probabilities which were stored before
            b=math.exp(b) #taking the exponent of those logarithmic probabilities
            for value1 in temp1:
                a+=value1 #adding up the logarithmic probabilities which were stored before
            a=math.exp(a) #taking the exponent of those logarithmic probabilities
            if b<=a: #comparing probability of a word based on language1 and language2 dictionaries
                unigram_lang1_accuracy+=1
            if b>a:
                unigram_lang2_accuracy+=1
            temp1=[]#resetting the variables so that probability could be calculated for the next word.
            temp2=[]
            a=0
            b=0
    return unigram_lang1_accuracy,unigram_lang2_accuracy

def lang1_vs_lang2_bigram(lang1,lang_dict1,lang_dict2):
    temp=''
    temp1=[]#holds probability value of a word based on language1 bigram probabilities.
    temp2=[]#holds probability value of a word based on language2 bigram probabilities.
    a=0
    b=0
    bigram_lang1_accuracy = 0#counters to hold accuracy values
    bigram_lang2_accuracy = 0
    for word in lang1:#for each word in test case
        if len(word)>1:#considering words greater than 1
            temp=list(nltk.ngrams(word.lower(),2))#creating bigrams of the test words.
            for char1 in temp:
                for char2 in lang_dict1.keys():
                    if char1==char2:#if char in test and language 1 dictionary matches 
                        temp1.append(lang_dict1[char1])
                    elif char1!=char2: #appending 0 if the characters in test and language1 dictionary are not matching
                        temp1.append(0)
                for char3 in lang_dict2.keys():
                    if char1==char3:#if char in test and language 2 dictionary matches 
                        temp2.append(lang_dict2[char1])
                    elif char1!=char3:#appending 0 if the characters in test and language2 dictionary are not matching
                        temp2.append(0)
            for value2 in temp2:
                b+=value2#adding up the logarithmic probabilities which were stored before
            b=math.exp(b)#taking the exponent of those logarithmic probabilities
            for value1 in temp1:
                a+=value1#adding up the logarithmic probabilities which were stored before
            a=math.exp(a)#taking the exponent of those logarithmic probabilities
            if b<=a:
                bigram_lang1_accuracy+=1
            elif b>a:
                bigram_lang2_accuracy+=1
        temp1=[]#resetting the variables so that probability could be calculated for the next word.
        temp2=[]
        b=0
        a=0
    return bigram_lang1_accuracy,bigram_lang2_accuracy
    
def lang1_vs_lang2_trigram(lang1,lang_dict1,lang_dict2):
    temp=''
    temp1=[]#holds probability value of a word based on language1 trigram probabilities.
    temp2=[]#holds probability value of a word based on language2 trigram probabilities.
    a=0 
    b=0
    trigram_lang1_accuracy = 0#counters to hold accuracy values
    trigram_lang2_accuracy = 0
    for word in lang1:#for each word in test case
        if len(word)>2:#considering words greater than 2
            temp=list(nltk.ngrams(word.lower(),3))
            for char1 in temp:
                for char2 in lang_dict1.keys():
                    if char1==char2:#if char in test and language 1 dictionary matches 
                        temp1.append(lang_dict1[char1])
                    if char1!=char2:#appending 0 if the characters in test and language1 dictionary are not matching
                        temp1.append(0)
                for char3 in lang_dict2.keys():
                    if char1==char3:#if char in test and language 2 dictionary matches 
                        temp2.append(lang_dict2[char1])
                    if char1!=char3:#appending 0 if the characters in test and language2 dictionary are not matching
                        temp2.append(0)
            for value2 in temp2:
                b+=value2#adding up the logarithmic probabilities which were stored before
            b=math.exp(b)#taking the exponent of those logarithmic probabilities
            for value1 in temp1:
                a+=value1#adding up the logarithmic probabilities which were stored before
            a=math.exp(a)#taking the exponent of those logarithmic probabilities
            if b<=a:
                trigram_lang1_accuracy+=1
            elif b>a:
                trigram_lang2_accuracy+=1
        temp1=[]#resetting the variables so that probability could be calculated for the next word.
        temp2=[] 
        a=0
        b=0
    return trigram_lang1_accuracy,trigram_lang2_accuracy

#################################################
#Function calls

english = udhr.raw('English-Latin1') 
english_train, english_dev = english[0:1000], english[1000:1100] 
english_test = udhr.words('English-Latin1')[0:1000] 
english_processed=text_processing(english_train)#preprocessing training set
english_char_count=char_count(english_processed)#frequency distribution of characters present in training set
eng_unigrams=list(nltk.ngrams(english_processed,1))#creating english unigrams
eng_unigram_count=unigram_count(english_char_count)#calculating english unigram probabilities
eng_bigrams = list(nltk.ngrams(english_processed,2))#creating english bigrams
eng_bigram_count,bigram_fd=bigram_count(eng_bigrams,english_char_count)#calculating english bigram probabilities
eng_trigrams=list(nltk.ngrams(english_processed,3))#creating english trigrams
eng_trigram_count=trigram_count(eng_trigrams,bigram_fd)#calculating english trigram probabilities

french = udhr.raw('French_Francais-Latin1') 
french_train, french_dev = french[0:1000], french[1000:1100] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]
french_processed=text_processing(french_train)#preprocessing training set
french_char_count=char_count(french_processed)#frequency distribution of characters present in training set
fr_unigrams=list(nltk.ngrams(french_processed,1))#creating french unigrams
fr_unigram_count=unigram_count(french_char_count)#calculating french unigram probabilities
fr_bigrams=list(nltk.ngrams(french_processed,2))#creating french bigrams
fr_bigram_count,bigram_fd=bigram_count(fr_bigrams,french_char_count)#calculating french bigram probabilities
fr_trigrams=list(nltk.ngrams(french_processed,3))#creating french trigrams
fr_trigram_count=trigram_count(fr_trigrams,bigram_fd)#calculating french trigram probabilities

spanish = udhr.raw('Spanish_Espanol-Latin1')
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]
spanish_processed=text_processing(spanish_train)#preprocessing training set
spanish_char_count=char_count(spanish_processed)#frequency distribution of characters present in training set
esp_unigrams=list(nltk.ngrams(spanish_processed,1))#creating spanish unigrams
esp_unigram_count=unigram_count(spanish_char_count)#calculating spanish unigram probabilities
esp_bigrams=list(nltk.ngrams(spanish_processed,2))#creating spanish bigrams
esp_bigram_count,bigram_fd=bigram_count(esp_bigrams,spanish_char_count)#calculating spanish bigram probabilities
esp_trigrams=list(nltk.ngrams(spanish_processed,3))#creating spanish trigrams
esp_trigram_count=trigram_count(esp_trigrams,bigram_fd)#calculating spanish trigram probabilities

italian = udhr.raw('Italian_Italiano-Latin1') 
italian_train, italian_dev = italian[0:1000], italian[1000:1100] 
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 
italian_processed=text_processing(italian_train)#preprocessing training set
italian_char_count=char_count(italian_processed)#frequency distribution of characters present in training set
ita_unigrams=list(nltk.ngrams(italian_processed,1))#creating italian unigrams
ita_unigram_count=unigram_count(italian_char_count)#calculating italian unigram probabilities
ita_bigrams=list(nltk.ngrams(italian_processed,2))#creating italian bigrams
ita_bigram_count,bigram_fd=bigram_count(ita_bigrams,italian_char_count)#calculating italian bigram probabilities
ita_trigrams=list(nltk.ngrams(italian_processed,3))#creating italian trigrams
ita_trigram_count=trigram_count(ita_trigrams,bigram_fd)#calculating italian trigram probabilities

a,b=lang1_vs_lang2_unigram(english_test,eng_unigram_count,fr_unigram_count)
print("The unigram accuracy of english model is ", (a/len(english_test))*100, " The accuracy of french model is ", (b/len(english_test))*100)
#
a,b=lang1_vs_lang2_bigram(english_test,eng_bigram_count,fr_bigram_count)
print("The bigram accuracy of english model is ", (a/len(english_test))*100, " The accuracy of french model is ", (b/len(english_test))*100)
#
a,b=lang1_vs_lang2_trigram(english_test,eng_trigram_count,fr_trigram_count)
print("The trigram accuracy of english model is ", (a/len(english_test))*100, " The accuracy of french model is ", (b/len(english_test))*100)
print("##############################################################################################")
a,b=lang1_vs_lang2_unigram(spanish_test,esp_unigram_count,ita_unigram_count)
print("The unigram accuracy of spanish model is ", (a/len(spanish_test))*100, " The accuracy of italian model is ", (b/len(spanish_test))*100)
#
a,b=lang1_vs_lang2_bigram(spanish_test,esp_bigram_count,ita_bigram_count)
print("The bigram accuracy of spanish model is ", (a/len(spanish_test))*100, " The accuracy of italian model is ", (b/len(spanish_test))*100)
#
a,b=lang1_vs_lang2_trigram(spanish_test,esp_trigram_count,ita_trigram_count)
print("The trigram accuracy of spanish model is ", (a/len(spanish_test))*100, " The accuracy of italian model is ", (b/len(spanish_test))*100)
