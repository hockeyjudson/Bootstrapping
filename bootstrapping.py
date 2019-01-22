#Function to merge multiple consecutive stars into a single star in a pattern
#Input masked pattern
def star_merger(pat_ls):
    k=0
    m=[]
    for i,j in enumerate(pat_ls):
        if j!="*" and k==0:
            m.append(j)
        elif j!="*" and k>0:
            m.append("*")
            m.append(j)
            k=0
        elif j=="*":
            if len(pat_ls)-1==i:
                m.append(j)
            k=k+1
    return m
#Function to apply star merger in all list of a nested list
#Input masked pattern
def base_pattern_creater(pat_ls):
    pat_ls=star_merger(pat_ls)
    return [star_merger(i) if i!="*" else "*" for i in pat_ls ]
#Function to fill maksed pattern with respect to unmask pattern
#input masked pattern and equivalent pattern
def pattern_filler1(a1,b1):
    #f=0
    m=0
    a=[]
    if len(a1)<=len(b1):
        for i,j in enumerate(a1):
            if a1[i]=="*":
                if i==0:
                    if len(a1)==1:
                        return b1
                    else:
                        for k in range(i,len(b1)):
                            if a1[i+k]==b1[k]:
                                m=k
                                break 
                            else:
                                a.append(b1[k])
                                m=k
                        if len(b1)-1==m:
                            return 1
                elif(len(a1)-1)==i:
                    if len(a)>=len(b1):
                        return a
                    elif(len(a)-1)==len(b1):
                        a.append(b1[-1])
                        return a
                    else:
                        for l in b1[m:]:
                            a.append(l)
                        return a
                else:
                    if len(b1)>m:
                        for k in range(m,len(b1)):
                            if a1[i+1]==b1[k]:
                                break 
                            else:
                                a.append(b1[k])
                                m=m+1
                            if len(b1)-1<=m:
                                return 1
            else:
                a.append(j)
                m=m+1
        else:
            return 1
def pattern_matcher(a1,b1):
    a=pattern_filler1(a1,b1)
    if a==1:
        return 1
    else:
        return a
#Function to implement pattern matcher
def base_pattern_matcher(pat,ip_list):
    patmod=pattern_matcher(pat,ip_list)
    #print(patmod,"----------",ip_list)
    if patmod==1:
        return 1
    else:
        if patmod==ip_list:
                return patmod
        else:
            return 1
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
#precision=TP/TP+FP
def pattern_score(tp,fp):
    return float(tp)/float(tp+fp)
#Function to find and match masked patterns in a list of patterns
#Input masked pattern,list of dependency paths
def pattern_match_collector(pat,ip_list):
    pat_col=[]
    for i in ip_list:
        s=base_pattern_matcher(pat,i)
        if isinstance(s,int):
            continue
        else:
            pat_col.append(s)
    return pat_col
