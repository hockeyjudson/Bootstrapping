from __future__ import division
import spacy
nlp=spacy.load("en")
#ip_list->list of text
#N->Ngrams N
def pos_dep_Ngrams(ip_list,N,s_words=1):
    pos_dct={}
    dep_dct={}
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    for i in ip_list:
        pos=[]
        dep=[]
        d=nlp(i.strip())
        if s_words==1:
            for j in d:
                if not j.is_punct:
                    if str(j.text).lower() not in sw:
                        pos.append(j.tag_)
                        dep.append(j.dep_)
        else:
            for j in d:
                if not j.is_punct:
                    pos.append(j.tag_)
                    dep.append(j.dep_)
        n=[]
        for k in [dep[p:p+N] for p in range(len(dep)-N+1)]:
            if k not in n:
                n.append(k)
        dep_dct[i]=n
        m=[]
        for k in [pos[p:p+N] for p in range(len(pos)-N+1)]:
            if k not in m:
                m.append(k)
        pos_dct[i]=m
    return [pos_dct,dep_dct]
#ip_txt->list of text
#pat->input pattern of pos tag 1d list of postag
#N-> Ngram N
def pos_match(ip_txt,pat,N,s_words=1):
    tx=[]
    deps=[]
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    for i in ip_txt:
        pos=[]
        txt=[]
        dep=[]
        d=nlp(i.strip())
        if s_words==1:
            for j in d:
                if str(j.text).lower() not in sw:
                    if not j.is_punct:
                        pos.append(j.tag_)
                        txt.append(j.text)
                        dep.append(j.dep_)
        else:
            for j in d:
                if not j.is_punct:
                    pos.append(j.tag_)
                    txt.append(j.text)
                    dep.append(j.dep_)
        for k in [(pos[p:p+N],[p,p+N]) for p in range(len(pos)-N+1)]:
            if k[0] == pat:
                tx.append(" ".join(txt[k[1][0]:k[1][1]]))
                deps.append(" ".join(dep[k[1][0]:k[1][1]]))
    return tx,deps
#ip_txt->list of text
#pat->input pattern of dep tag 1d list of postag
#N-> Ngram N
def dep_match(ip_txt,pat,N,s_words=1):
    tx=[]
    deps=[]
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    for i in ip_txt:
        pos=[]
        txt=[]
        dep=[]
        d=nlp(i.strip())
        if s_words==1:
            for j in d:
                if str(j.text).lower() not in sw:
                    if not j.is_punct:
                        pos.append(j.tag_)
                        txt.append(j.text)
                        dep.append(j.dep_)
        else:
            for j in d:
                if not j.is_punct:
                    pos.append(j.tag_)
                    txt.append(j.text)
                    dep.append(j.dep_)
        for k in [(dep[p:p+N],[p,p+N]) for p in range(len(dep)-N+1)]:
            if k[0] == pat:
                tx.append(" ".join(txt[k[1][0]:k[1][1]]))
                deps.append(" ".join(dep[k[1][0]:k[1][1]]))
    return tx,deps
def merge_print(t,s):
    for i in range(len(s)):
        s1=t[i].split()
        s2=s[i].split()
        print(" ".join([s1[k]+"/"+s2[k] for k in range(len(s1))]))
#input ip_list->2d neseted list
def frequent_patterns(ip_list):
    import numpy as np
    ls={}
    for i in ip_list:
        for j in i:
            j=",".join(j)
            if j not in ls:
                ls[j]=[j.split(","),1]
            else:
                ls[j][1]=ls[j][1]+1
    f=[j[1] for i,j in ls.items()]
    f_i=np.argsort(f)
    f_i=f_i[::-1]
    f_l=list(ls)
    freq=[ls[f_l[i]] for i in f_i]
    return freq
#input f list->1d strings
#swords-> int to use stopwords or not for word counting
#output frequency of words in the string
def word_count(f,swords=1):
    from nltk.corpus import stopwords
    if swords==1:
        fc=[j for i in f for j in i.lower().strip().split() if j not in stopwords.words('english') ]
    else:
        fc=[j for i in f for j in i.lower().strip().split() ]
    from collections import Counter
    return Counter(fc).most_common()
#input sentence as string
#input s_words as boolean
#output list->[word]/[dep_tag]/[pos_tag]
def triplet_form(sent,s_words=False):
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    doc=nlp(sent)
    ls=[]
    if s_words:
        for i in doc:
            if i.lower_ not in sw and i.dep_!="punct":
                s=i.text+"/"+i.dep_+"/"+i.tag_
                ls.append(s)
    else:
        for i in doc:
            s=i.text+"/"+i.dep_+"/"+i.tag_
            ls.append(s)
    return ls
#input sent as string
#input s_words as boolean
#output list(word,dep,pos tag)
def triples(sent,s_words=False):
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    doc=nlp(sent)
    word=[]
    dep=[]
    tag=[]
    if s_words:
        for i in doc:
            if i.lower_ not in sw and i.dep_!="punct":
                word.append(i.text)
                dep.append(i.dep_)
                tag.append(i.tag_)
    else:
        for i in doc:
            word.append(i.text)
            dep.append(i.dep_)
            tag.append(i.tag_)
    return [word,dep,tag]
#input triplet form list
#output three list(words,dep,pos)
def parse_triplet(lst):
    word=[]
    dep=[]
    pos=[]
    for i in lst:
        s=i.split("/")
        word.append(s[0])
        dep.append(s[1])
        pos.append(s[2])
    return word,dep,pos
#input s->1d list or string
#input t->1d list or string
#output jaro distance in float
def jaro(s, t):
    s_len = len(s)
    t_len = len(t)
 
    if s_len == 0 and t_len == 0:
        return 1
 
    match_distance = (max(s_len, t_len) // 2) - 1
 
    s_matches = [False] * s_len
    t_matches = [False] * t_len
 
    matches = 0
    transpositions = 0
 
    for i in range(s_len):
        start = max(0, i-match_distance)
        end = min(i+match_distance+1, t_len)
 
        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
 
    if matches == 0:
        return 0
 
    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1
 
    return ((matches / s_len) +(matches / t_len) +((matches - transpositions/2) / matches)) / 3
#input s->1d list or string
#input t->1d list or string
#input l-> integer of prfix
#input p->float scaling factor must not exceed .25 according to paper
#output jaro distance in float
def jaro_winkler(s,t,l=3,p=0.1):
    return jaro(s,t)+(l*p)*(1-jaro(s,t))
#input ref_pattern-> 1d list
#input list_pattern-> 2d list list of pattern
#output dictionary with  keys{avg:average value,min:minimum value,max:maximum value }
def avg_jaro_score(ref_pattern,list_pattern):
    score=[pe.jaro(ref_pattern,i) for i in list_pattern]
    score_dict={}
    score_dict['avg']=sum(score)/len(score)
    score_dict['min']=min(score)
    score_dict['max']=max(score)
    return score_dict
#input indx -> int
#input window_size -> int
#input length -> int
#output list[start_index,end_index]
def align_window(indx,window_size,length):
    if window_size>=length:
        return[0,length]
    else:
        l=r=window_size//2
        if indx<l:
            r=r+(l-indx)+indx
            #l=0
            return [0,r+1]
        elif length-(indx+1)<r:
            l=l+(r-(length-(indx+1)))
            l=indx-l
            return [l,length]
        else:
            return[indx-l,indx+r+1]
#input sent ->String
#input tag-> 1d list
#input window_size-> int 
#input stop_words->False
#ouput nested list
def flex_window_patEx(sent,tag_list,window_size=5,stop_words=False):
    tptfrm=triples(sent,stop_words)
    word=tptfrm[0]
    dep=tptfrm[1]
    tag=tptfrm[2]
    ret_list=[]
    for i in tag_list:
        for j,k in enumerate(dep):
            if k==i and tag[j]!='CD':
                ind=align_window(j,window_size,len(dep))
                ret_list.append(tag[ind[0]:ind[1]])
    return ret_list
#input iplist->list or nested list of patterns
#output dictionary number of elements
def counting_elements(ip_list):
    import itertools
    sl=list(itertools.chain(*ip_list))
    count_dict={}
    for i in sl:
        if i not in count_dict:
            count_dict[i]=1
        else:
            count_dict[i]=count_dict[i]+1
    return count_dict
#input x->int
#input y->int
#output int
#note 2.718 refers euler constant 'e'
def scoring(x,y):
    return 2.718**x/(2.718**x+2.718**y)