from __future__ import division

import pandas as pd
import nltk


trainvar = pd.read_csv("../data/training_variants")
traintext = pd.read_csv("../data/training_text", sep="\|\|",
                        header=None, skiprows=1, names=["ID", "Text"])

datos = trainvar.merge(traintext, on='ID')

stopwords = nltk.corpus.stopwords.words('english')


def wordFilter(excluded, row):
    filtered = [w for w in row if w not in excluded]
    return filtered


def lowerCaseArray(wordrow):
    lowercased = [word.lower() for word in wordrow]
    return lowercased


porter = nltk.PorterStemmer()


def wordStemmer(wordrow):
    stemmed = [porter.stem(word) for word in wordrow]
    return stemmed


def text_processing(df, clase):
    """Por cada clase crea un diccionario (data) con dos elementos: una matriz,
     que es una lista de listas donde cada sublista es uno de los textos
     dividido por palabras, y una lista con todas las palabras de la clase.
     De esta forma, puedo limpiar cada uno de los textos (minusculas, sufijos,
     prefijos...) e identificar las palabras que solo se repiten una vez por
     cada texto para eliminarlas.
    """
    data = {'matrix': [], 'all': []}
    textos = df.loc[df['Class'] == clase]['Text']
    interWordMatrix = []
    interWordList = []

    for i in textos:
        raw = i.decode("utf8")   # pasar el texto a utf8
        tokens = nltk.word_tokenize(raw)  # dividir texto en palabras
        wordrow_lowercased = lowerCaseArray(tokens)  # pasar a minusculas
        # eliminar determinantes y demas
        wordrow_nostopwords = wordFilter(stopwords, wordrow_lowercased)
        wordrow_stemmed = wordStemmer(wordrow_nostopwords)  # quitar sufijos, afijos
        # lista con todas las palabras de los textos de la clase
        interWordList.extend(wordrow_stemmed)
        interWordMatrix.append(wordrow_stemmed)  # lista de listas con cada texto

    wordfreqs = nltk.FreqDist(interWordList)  # calculo la frecuencia de cada palabra en el global
    hapaxes = wordfreqs.hapaxes()  # identifico las palabras que solo se repiten una vez en toda la clase
    for wordvector in interWordMatrix:
        # filtro de las palabras que solo se repiten 1 vez
        wordvector_nohapexes = wordFilter(hapaxes, wordvector)
        data['matrix'].append(wordvector_nohapexes)  # relleno la matriz con los textos ya limpios
        data['all'].extend(wordvector_nohapexes)  # relleno la lista con los textos tambien limpios

    print ('---')

    return data


alldata = {}
for clase in datos['Class'].unique():
    alldata[clase] = text_processing(datos, clase)
