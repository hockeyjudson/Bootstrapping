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
    score=[jaro(ref_pattern,i) for i in list_pattern]
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
    retdict={}
    for i in tag_list:
        for j,k in enumerate(dep):
            if k==i and tag[j]!='CD':
                if i not in retdict:
                    ind=align_window(j,window_size,len(dep))
                    retdict[i]=[tag[ind[0]:ind[1]]]
                else:
                    ind=align_window(j,window_size,len(dep))
                    retdict[i].append(tag[ind[0]:ind[1]])
    return retdict
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
def pattern_tagger(file_name,pat,tag,window_size=6,stop_words=True):
    import os
    pat_collect=[]
    new_list=[]
    flag=0
    sent_list=open(file_name,"r").readlines()
    for i in sent_list:
        ls=triples(i.strip(),stop_words)
        print(ls)
        max=0.0
        dt={"pat":[],"score":0.0}
        for j in range(len(ls[1])-window_size+1):
            sc=jaro(ls[1][j:j+window_size],pat)
            #print(sc,ls[1][j:j+window_size])
            if sc>max:
                max=sc
                dt["pat"]=ls[1][j:j+window_size]
                dt["score"]=sc
                print(dt)
        if dt["score"]==1.0:
            flag=1
            i=i.strip()+" "+tag+" \n"
            new_list.append(i)
        elif .87<dt["score"]<1.0:
            pat_collect.append(dt)
            new_list.append(i)
        else:
            new_list.append(i)
    """if flag==1:
        os.remove(file_name)
        f=open(file_name,"a")
        for i in new_list:
            f.write(i)
        f.close()"""
    return [flag,new_list,pat_collect]
#input sd->list(1d) of patterns to be count
#input c->count dictionary stored in pickle file:count_element.pickle
#input tag->string (dictionary key such as arguments.pickle,facts.pickle,identify.pickle,ratio.pickle,decision.pickle)
#output dict with element and its counts
def element_counter(sd,c,tag):
    dt={}
    for i in sd:
        dt[i]={}
    for i in sd:
        dt[i]["x"]=0
        dt[i]["y"]=0
        for j in c.keys():
            if j==tag:
                if i not in c[j]:
                    dt[i]["x"]=0
                else:
                    dt[i]["x"]=c[j][i]
            else:
                if i not in c[j]:
                    dt[i]["y"]=dt[i]["y"]+0
                else:
                    dt[i]["y"]=dt[i]["y"]+c[j][i]
    return dt
#input dict_ele_count->dict input the value from element_counter function
#input element_counter(element_counter(sd[0],c,"arguments.pickle"))
#output list[dict->element with values,string->element with the lowest score
def score_min_val(dict_ele_count):
    score={}
    for i,j in dict_ele_count.items():
        score[i]=scoring(*list(j.values()))
    return [score,min(score, key=lambda k: score[k])]
#input element->string tag going to match in the initial position of the list
#input pat_list->list list of patterns
#output list->id list which return dictionary of elements and its count and elemant with the maximum count
def count_intial(element,pat_list):
    dct={}
    for i in pat_list:
        if element==i[1]:
            if i[0] not in dct:
                dct[i[0]]=1
            else:
                dct[i[0]]=dct[i[0]]+1
    if dct=={}:
        return 0
    else:
        return [dct,max(dct, key=lambda k: dct[k])]
#input element->string tag going to match in the final position of the list
#input pat_list->list list of patterns
#output list->id list which return dictionary of elements and its count and elemant with the maximum count
def count_final(element,pat_list):
    dct={}
    for i in pat_list:
        if i[-2]==element:
            if i[-1] not in dct:
                dct[i[-1]]=1
            else:
                dct[i[-1]]=dct[i[-1]]+1
    if dct=={}:
        return 0
    else:
        return [dct,max(dct, key=lambda k: dct[k])]  
#input elements->list of tag going 
#input pat_list->list list of patterns
#output list->id list which return dictionary of elements and its count and elemant with the maximum count
def count_pair(elements,pat_list):
    dct={}
    for i in pat_list:
        for j,k in enumerate(i[:-2]):
            if k==elements[0] and elements[1]==i[j+2]:
                if i[j+1] not in dct:
                    dct[i[j+1]]=1
                else:
                    dct[i[j+1]]=dct[i[j+1]]+1
    if dct=={}:
        return 0
    else:
        return [dct,max(dct, key=lambda k: dct[k])]
#input pat->list(1d list of elements)
#input pat_list->list of patterns
#input tag->string (dictionary key such as arguments.pickle,facts.pickle,identify.pickle,ratio.pickle,decision.pickle)
#input pat_list1->dict elements with count
#output modified pattern or else string with neglected pattern
def pattern_modify(pat,pat_list,tag,pat_list1):
    dct=element_counter(pat,pat_list1,tag)
    print(dct)
    dct_min=score_min_val(dct)
    print(dct_min)
    ele=[]
    for i,j in enumerate(pat):
        if dct_min[1]==pat[i]:
            if i==0:
                ele.append([count_intial(pat[1],pat_list),i])
            elif i==len(pat)-1:
                ele.append([count_final(pat[-2],pat_list),i])
            else:
                ele.append([count_pair([pat[i-1],pat[i+1]],pat_list),i])
    print(ele)
    if 0 in ele:
        return "pattern neglected"
    else:
        for i in ele:
            pat[i[1]]=i[0][1]
    return pat