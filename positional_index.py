from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import string
import re


class PositionalIndex:

    def __init__(self):
        self.__tokens = []
        self.__terms = []
        self.__model = {}
        self.__model_df = pd.DataFrame()

    def __pre_process(self, doc) -> str:
        self.__tokens.clear()
        self.__terms.clear()
        file = open(f'docs/{doc}.txt', 'r')
        for line in file.readlines():
            self.__tokens.extend(line.lower().split())
        self.__tokens = PositionalIndex.remove_punc(self.__tokens)
        self.__terms = self.__tokens
        self.__tokens = PositionalIndex.stem(self.__tokens)
        self.__terms = PositionalIndex.remove_stop_words(self.__terms)
        self.__terms = PositionalIndex.stem(self.__terms)
        self.__terms = set(self.__terms)

    @staticmethod
    def remove_punc(tokens) -> list:
        for i in tokens:
            if i in string.punctuation:
                tokens.remove(i)
        return [i[:-1] if i[-1] in string.punctuation else i for i in tokens]

    @staticmethod
    def remove_stop_words(tokens) -> list:
        return [i for i in tokens if i not in stopwords.words('english')]

    @staticmethod
    def stem(tokens) -> list:
        ps = PorterStemmer()
        return [ps.stem(i) for i in tokens]

    def build(self, list_of_docs) -> list:
        indexes = []
        for doc in list_of_docs:
            self.__pre_process(doc)
            for term in self.__terms:
                for i in range(len(self.__tokens)):
                    if term == self.__tokens[i]:
                        indexes.append(i)
                if term in self.__model:
                    self.__model[term][re.findall(r'\d{1,2}', doc)[0]] = indexes.copy()
                else:
                    self.__model[term] = {re.findall(r'\d{1,2}', doc)[0]: indexes.copy()}
                indexes.clear()
        doc_freq = [len(self.__model[i]) for i in self.__model]
        doc_index = [self.__model[i] for i in self.__model]
        self.__model_df = pd.DataFrame({
            'term': list(self.__model),
            'doc_freq': doc_freq,
            'docs and indexes': doc_index
        })
        self.__model_df = self.__model_df.sort_values('term')
        self.__model_df = self.__model_df.set_index(pd.Index([i for i in range(len(self.__model_df))]))

    def show(self):
        print('-' * 50 + '  Positional Index  ' + '-' * 50)
        print(self.__model_df)

    def get_model(self):
        return self.__model_df

    def enter_query(self, query) -> str:
        query_terms = PositionalIndex.remove_punc(query.lower().split())
        query_terms = PositionalIndex.remove_stop_words(query_terms)
        query_terms = PositionalIndex.stem(query_terms)

        if query_terms == []:
            print('The entered query doesn\'t exist 😮')
            return

        for i in query_terms:
            if i not in list(self.__model_df['term']):
                print('The entered query doesn\'t exist 😮')
                return

        docs_indexes = {}
        for i in query_terms:
            docs_indexes[i] = list(self.__model_df.loc[self.__model_df['term'] == i]['docs and indexes'])[0]
        if len(query_terms) == 1:
            print('-' * 50 + '  Result Matrix  ' + '-' * 50)
            print(pd.DataFrame({
                'doc': [i for i in list(docs_indexes[query_terms[0]])],
                'indexes': [docs_indexes[query_terms[0]][i] for i in list(docs_indexes[query_terms[0]])]
            }).sort_values('doc'))
            return

        i = 0
        docs = set(docs_indexes[query_terms[0]])
        while i < (len(query_terms) - 1):
            docs.intersection_update(set(docs_indexes[query_terms[i + 1]]))
            i += 1

        if len(docs) == 0:
            print('The entered query doesn\'t exist 😮')
            return

        result = {}
        for doc in docs:
            i = 0
            indexes = set([j for j in docs_indexes[query_terms[i]][doc]])
            while i < (len(query_terms) - 1):
                indexes = set([j + 1 for j in indexes])
                indexes.intersection_update(set(docs_indexes[query_terms[i + 1]][doc]))
                i += 1
            result[doc] = sorted(list(indexes.copy()))
        delete = []
        for i in result:
            if result[i] == []:
                delete.append(i)
        for i in delete:
            del result[i]

        if len(result) == 0:
            print('The entered query doesn\'t exist 😮')
            return

        for doc in docs:
            if doc in result.keys():
                indexes = []
                for index in result[doc]:
                    temp_indexes = []
                    temp_indexes.append(index)
                    for i in range(1, len(query_terms)):
                        temp_indexes.append(index - i)
                    indexes.append(sorted(temp_indexes.copy()))
                result[doc] = indexes.copy()

        result = pd.DataFrame({
            'doc': [i for i in result],
            'indexes': [result[i] for i in result]
        }).sort_values('doc')
        print('-' * 50 + '  Result Matrix  ' + '-' * 50)
        print(result)
