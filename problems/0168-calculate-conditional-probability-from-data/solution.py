from collections import defaultdict
def conditional_probability(data, x, y):
    mpp = defaultdict(list)
    for X,Y in data:
        mpp[X].append(Y)

    cnt = 0
    n = 0
    for val in mpp[x]:
        if val==y:
            cnt +=1
        n+=1
    try:
        res = cnt/n
    except:
        res = 0
    return round(res,4)