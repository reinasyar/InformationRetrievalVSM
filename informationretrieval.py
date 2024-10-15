#%%
from vectorization import vectorization
import os

#%%
class informationretrieval:
    def __init__(self):
        path = './documents'
        self.docs = os.listdir(path)
        # print(docs, len(docs))
        
        self.vtr = vectorization()
        
        self.l_documents = []
        for document in self.docs:
            temp = './documents/'+document
            string = self.vtr.text_to_string(temp)
            self.l_documents.append(string)
            
        # print(len(self.l_documents))
        self.vocabulary = self.vtr.vocabulary_maker(self.l_documents)
        # print(len(self.vocabulary))
        
        self.tfidf = self.vtr.tfidf_maker(self.vocabulary, self.l_documents, self.docs)
        
    
    
    def run(self):
        while 1:
            print("type 'close' without '' for stop searching")
            t_query = input("what are you looking for? ")
            query = self.vtr.remove_symbol(t_query)
            if(query=="close"):
                print("thank you")
                break
            
            v_query = self.vtr.vector_query_maker(query, self.vocabulary)
            
            sim = {}
            
            for doc in self.tfidf.columns:
                doc_vector = dict(zip(self.vocabulary, self.tfidf[doc].tolist()))
                similar = self.vtr.cosine_similarity(v_query, doc_vector)
                if similar>0:
                    sim[doc] = similar
            sorted_sim = dict(sorted(sim.items(), key=lambda item: item[1], reverse=True))
        
            tops = list(sorted_sim.keys())[:3]
            
            print(" ")
            if tops:
                print("the document you search: ")
                for i, top in enumerate(tops):
                    print(f"{i+1}. {top}")
            
            else:
                print("there is no document for the query")
                
            print(" ")
            print(" ")
            
            
        
        
#%%
ir = informationretrieval()
ir.run()
# a = []
# for i in range(21):
#     a.append(i)
# temp_data_dict = dict(zip(df.columns, a))
# df.loc[0] = temp_data_dict
#%%