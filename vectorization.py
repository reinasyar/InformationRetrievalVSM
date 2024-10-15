#%%
import math
import re
import pandas as pd

#%%
class vectorization:
    def text_to_string(self, path):
        # baca txt file ke string
        with open(path, "r") as file:
            lines = file.readlines()
        content = "".join(lines)
        
        # remove simbol-simbol dan ubah tiap huruf ke lower case
        new_content = self.remove_symbol(content)
  
        return new_content
    
    def remove_symbol(self, string):
        pattern = r"[^\w\s]" 
        new_string = re.sub(pattern, "", string.lower())
        
        return new_string
    
    def tokenization(self, string):
        # bikin token dari string
        tokens = string.split()
        return tokens
    
    def vocabulary_maker(self, l_documents):
        # bikin vocabs dari list of string
        vocabs = set()
        for doc in l_documents:
            tokens = self.tokenization(doc)
            vocabs.update(tokens)
            
        return vocabs
        
    def tfidf_maker(self, vocabulary, l_documents,  n_docs):
        tfidf = pd.DataFrame(columns=n_docs, dtype=float)
        
        len_d = len(n_docs)
        for i, term in enumerate(vocabulary):
            temp_tf = []
            c_df = 0
            
            for doc in l_documents:
                c_tf = 0
                for tkn in self.tokenization(doc):
                    if(tkn==term):
                        c_tf=c_tf+1
                temp_tf.append(c_tf)
                if c_tf>0:
                    c_df = c_df+1
                    
            idf = math.log(len_d/c_df)
            tfidf_t = []
            for tf in temp_tf:
                temp = tf * idf
                tfidf_t.append(temp)
                
            temp_dict = dict(zip(tfidf.columns, tfidf_t))
            tfidf.loc[i] = temp_dict
        
        return tfidf
    
    def vector_query_maker(self, query, vocabulary):
        v_query = {term: 0 for term in vocabulary}
        tokens = self.tokenization(query)
        for term in tokens:
            if term in v_query:
                v_query[term] = 1
        return v_query
    
    def cosine_similarity(self, v1, v2):
        dot = 0
        t_ed_v2 = 0
        for term, value in v1.items():
            if term in v2:
                dot += value * v2[term]
            t_ed_v2 += (v2.get(term, 0) ** 2)
        ed_v2 = math.sqrt(t_ed_v2)
        
        if(ed_v2>0):
            t_ed_v1 = 0
            for value in v1.values():
                t_ed_v1 += value ** 2
            
            ed_v1 = math.sqrt(t_ed_v1)
            
            if ed_v1>0:
                return dot/(ed_v2 * ed_v1)
            else:
                return 0
        
        else:
            return 0
    
#%%
    
    
# vtr = vectorization()

# string = vtr.text_to_string('./documents/Blockchain.txt')
# print(string)

# tkn = vtr.tokenization(string)
# print(tkn, len(tkn))

# test = set()
# test.update(tkn)
# print(len(test))