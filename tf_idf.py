from positional_index import PositionalIndex
from math import log10
from math import sqrt
import pandas as pd
import numpy as np
import re


class TF_IDF:
    def __init__(self, model, docs):
        self.__model = model
        self.__docs = docs
        self.__tf = None
        self.__tf_wt = None
        self.__idf = None
        self.__tf_idf = None
        self.__normalized_tf_idf = None
        self.__generate()

    def __generate(self):
        self.__tf_create()
        self.__tf_wt_create()
        self.__idf_create()
        self.__tf_idf_create()
        self.__normalized_tf_idf_create()

    def __tf_create(self):
        self.__tf = pd.DataFrame({
            'term': self.__model['term'],
        })
        for doc in self.__docs:
            tf = []
            for i in self.__model['docs and indexes']:
                if re.findall(r'\d{1,2}', doc)[0] in i:
                    tf.append(len(i[re.findall(r'\d{1,2}', doc)[0]]))
                else:
                    tf.append(0)
            self.__tf[f'tf {doc}'] = tf.copy()

    def __tf_wt_create(self):
        self.__tf_wt = pd.DataFrame({
            'term': self.__model['term'],
        })
        for doc in self.__docs:
            tf = list(self.__tf[f'tf {doc}'])
            tf_wt = [0 if i == 0 else round((1 + log10(i)), 2) for i in tf]
            self.__tf_wt[f'tf_wt {doc}'] = tf_wt

    def __idf_create(self):
        df = list(self.__model['doc_freq'])
        idf = [round(log10(len(self.__docs) / int(i)), 2) for i in df]
        self.__idf = pd.DataFrame({
            'term': self.__model['term'],
            'doc_freq': self.__model['doc_freq'],
            'idf': idf
        })

    def __tf_idf_create(self):
        self.__tf_idf = pd.DataFrame({
            'term': self.__model['term'],
        })
        for doc in self.__docs:
            tf_wt = list(self.__tf_wt[f'tf_wt {doc}'])
            tf_idf = [round(i * j, 2) for i, j in zip(tf_wt, list(self.__idf['idf']))]
            self.__tf_idf[f'tf_idf {doc}'] = tf_idf

    def __normalized_tf_idf_create(self):
        self.__normalized_tf_idf = pd.DataFrame({
            'term': self.__model['term'],
        })
        for doc in self.__docs:
            tf_idf = list(self.__tf_idf[f'tf_idf {doc}'])
            length = [i ** 2 for i in tf_idf]
            length = round(sqrt(sum(length)), 2)
            self.__normalized_tf_idf[f'norm_{doc}'] = [round(i / length, 2) for i in tf_idf]

    def show(self):
        print('-' * 50 + '  TF Matrix  ' + '-' * 50)
        print(self.__tf)
        print('-' * 50 + '  TF_Weight Matrix  ' + '-' * 50)
        print(self.__tf_wt)
        print('-' * 50 + '  IDF Matrix  ' + '-' * 50)
        print(self.__idf)
        print('-' * 50 + '  TF_IDF Matrix  ' + '-' * 50)
        print(self.__tf_idf)
        print('-' * 50 + '  Normalized TF_IDF Matrix  ' + '-' * 50)
        print(self.__normalized_tf_idf)

    def enter_query(self, query):
        query_terms = PositionalIndex.remove_punc(query.lower().split())
        query_terms = PositionalIndex.remove_stop_words(query_terms)
        query_terms = PositionalIndex.stem(query_terms)

        if query_terms == []:
            print('The entered query doesn\'t exist 😮')
            return

        query = pd.DataFrame({
            'term': self.__model['term'],
            'tf': [0 for i in self.__model['term']],
            'tf_wt': [0 for i in self.__model['term']],
            'df': [i for i in self.__model['doc_freq']],
            'idf': [i for i in self.__idf['idf']],
            'tf_idf': [0 for i in self.__model['term']],
            'norm_query': [0 for i in self.__model['term']],

        })
        for i in query_terms:
            if i in list(self.__model['term']):
                query.loc[query.term == i, 'tf'] = query_terms.count(i)
        query['tf_wt'] = [0 if i == 0 else round((1 + log10(i)), 2) for i in list(query['tf'])]
        query['tf_idf'] = [round(i * j, 2) for i, j in zip(list(query['tf_wt']), list(query['idf']))]
        length = [i ** 2 for i in list(query['tf_idf'])]
        length = round(sqrt(sum(length)), 2)
        query['norm_query'] = [round(i / length, 2) for i in list(query['tf_idf'])]
        result = {}
        for doc in self.__docs:
            result[f'(query, {doc})'] = round(np.array(query['norm_query']).dot(
                np.array(self.__normalized_tf_idf[f'norm_{doc}'])), 2)
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        print('-' * 50 + '  Query Matrix  ' + '-' * 50)
        print(query)
        print('-' * 50 + '  Similarity(Query, Doc)  ' + '-' * 50)
        result = pd.DataFrame({
            "": list(result),
            "similarity": [result[i] for i in result]
        })
        print(result)
