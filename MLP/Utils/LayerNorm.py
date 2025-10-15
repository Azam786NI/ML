import numpy as np
def LN(lis):
    mean=np.average(lis)
    print("\nmean ",mean)

    variance=0

    for i in lis:
        deviation=i-mean    
        variance+=deviation**2

        print("deviation ",deviation)

    variance=variance/len(lis)
    print("nvariance ",variance)

    standard_deviation=variance**(1/2)
    print("standard_deviation ",standard_deviation,end="\n\n")

    normalized_list=[]
    for i in lis:
        z=(i-mean)/standard_deviation
        normalized_list.append(z)


    return [normalized_list]
