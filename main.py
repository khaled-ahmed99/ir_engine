from positional_index import PositionalIndex
from tf_idf import TF_IDF

model = PositionalIndex()
docs = ['doc1', 'doc2', 'doc3']
# , 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10'

model.build(docs)
model.show()

tf_idf = TF_IDF(model.get_model(), docs)
tf_idf.show()

query = input('enter query or type "eot" to exit')
while query.lower() != 'eot':
    model.enter_query(query)
    tf_idf.enter_query(query)
    query = input('enter query or type "eot" to exit')
else:
    print('❤ thanks for using our IR System ❤')
