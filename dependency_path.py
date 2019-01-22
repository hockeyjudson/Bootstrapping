import spacy,networkx
nlp=spacy.load('en_core_web_sm')
#Input String
def remove_stop_punct(s):
        from nltk.corpus import stopwords
        stopwords=stopwords.words('english')
        doc=nlp(s)
        #k=sorted([".","DT","TO","CC","IN"])
        #st=" ".join(t.text for t in doc if (t.pos_ not in ["PUNCT"] )and(t.text.lower() not in stopwords))
        st=" ".join(t.text for t in doc if (t.pos_ not in ["PUNCT"] ))
        return st.strip()
#Input String
#Output Dictionary and root of the parse tree
def dict_conv(st):
    doc=nlp(st)
    root=[]
    for i in doc:
        if i.dep_=="ROOT":
            root.append(i.i)
    d={tok.i:[tok.text,tok.dep_,tok.head.text,tok.tag_,[child.i for child in tok.children]] for tok in doc}
    return [d,root]
#Function to find all dependency paths of a given sentences 
def dfs(sd, start, end = "$$"):   
    h=networkx.Graph(sd)
    ptr=[]
    for pth in networkx.all_simple_paths(h,start,"$$"):
        ptr.append(pth[:-1])
    return ptr
#Function to parse all dependency paths into the tree
def parse_depend(dt,root):
    sd=dict()
    sd["$$"]=[]
    for i,j in dt.items():
        if j[4]==[]:
            sd[i]=["$$"]
        else:
            sd[i]=[k for k in j[4]]
    ml=[]
    for i in root:
        ml.extend(dfs(sd,i,"$$"))
    #print ml
    mpos=[[dt[j][3] for j in i]for i in ml]
    mdep=[[dt[j][1] for j in i]for i in ml]
    return mdep,mpos
#input nested string list output string
def To_str(nlst):
    st="("
    for i in nlst:
        st=st+"("
        st=st+",".join(map(str,i))
        st=st+"),"
    return st[:-1]+")"
#find unique nested list
#input ip_list->nested list
#output uniq_list->nested list
def uniq_list(ip_list):
    uniq_list=[ip_list[0]]
    for i in ip_list:
        k=0
        for j in uniq_list:
            if j==i:
                k=0
                break
            else:
                k=1
        if k==1:
            uniq_list.append(i)
    return uniq_list
