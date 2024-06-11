# -*- coding: utf-8 -*-
"""
module pln_ner.py
--------------------
Named Entuty Recognition classifier parser.
"""
from __future__ import unicode_literals, print_function
import os
import spacy
from spacy import displacy
from spacy.symbols import nsubj, VERB
import pandas as pd
import random
import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


class Ner(object):

    def __init__(self):
        # print(os.getcwd())
        self.nlp = spacy.load(
            'nlp/robo.ner'
        ) # a pasta deve estar descompactada

    def processSentence(self, sentence, _debug = False):
        doc = self.nlp(sentence)

        who = -1
        raiz = -1
        perVerb = -1

        for token in doc:
            if _debug:
                print(token.i, token.text.ljust(10), token.pos_.ljust(10), 
                      token.tag_.ljust(10), token.dep_.ljust(10), 
                      token.head.text.ljust(10), token.head.pos_.ljust(10), 
                      [child for child in token.children])
            
            if token.dep_=="nsubj":
                who = token.i
            elif token.dep_=="ROOT":
                raiz = token.i
            elif token.pos_=="VERB":
                if perVerb == -1:
                    perVerb = token.tag_.split('|')[3][0]

        if raiz != -1:
            place = -1
            while place==-1:
                for token in doc[raiz].rights:
                    #print (token, end=" ")
                    if token.dep_ in ["nmod", "obl"]:
                        place = token.i
                        break
                if place==-1:
                    if list(doc[raiz].rights)==[]:
                        break
                    for token in doc[raiz].rights:
                        raiz = token.i
                        break
            if place!=-1:
                lcomp = []
                rcomp = []
                if place!=-1:
                    for token in doc[place].lefts:
                        if token.dep_=="amod":
                            lcomp.append(token.i)
                    for token in doc[place].rights:
                        if token.dep_=="amod":
                            rcomp.append(token.i)            
        #print ([who, place, lcomp, rcomp])    
        if who==-1:
            if perVerb=="3":
                ent = "Robô"
            elif perVerb=="1":
                ent = "Pessoa"
            else:
                ent = "?"
        elif doc[who].text in ["você", "tu", "robô"]:
            ent = "Robô"
        elif (doc[who].text in ["eu"]):        
            ent = "Pessoa"
        else:
            ent = "?" 

        if place==-1:
            fullPlace = "?"
        else:
            fullPlace = ""
            for l in lcomp:
                fullPlace = fullPlace + doc[l].text + " "
            fullPlace += doc[place].text
            for r in rcomp:
                fullPlace = fullPlace + " " + doc[r].text
        
        return [ent, fullPlace]

    def processEntities(self, sentence, _debug = False):
        #enganando o modelo
        sentence = sentence.replace("estás ", "está ")

        doc = self.nlp(sentence)
        posFound = False
        fullPlace = ""
        who = ""
        for ent in doc.ents:
            if _debug:
                print(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.label_ == "PLA":
                fullPlace = ent.text
            elif ent.label_ == "POS":
                posFound = True
            elif ent.label_ == "WHO":
                who = ent.text
        
        if who.lower() in ["você", "tu", "robô"]:
            ent = "Robô"
        elif who.lower() in ["eu"]:
            ent = "Pessoa"
        else:
            ent = "?" 

        if _debug:
            print(ent, fullPlace, posFound)

        out = self.processSentence(sentence)

        if fullPlace!="" and ent == "?":
            ent = out[0]
            posFound = posFound and ent!="?"
        elif fullPlace=="" and ent == "?":
            ent = out[0]
            fullPlace = out[1]
            posFound = posFound and ent!="?" and fullPlace != ""

        return (ent, fullPlace, posFound)
    
    def __call__(self, sentence):
        """ Named Entities Recognition call.
        
        Named Entities Recognition in a sentence given as input.
        :param sentence: A text sentence.
        :type sentence: str
        """
        entity, place, found = self.processEntities(sentence)
        return entity, place, found

