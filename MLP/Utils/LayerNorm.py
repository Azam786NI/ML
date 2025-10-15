import numpy as np
def LN(lis):
    mean=np.average(lis)
    print("mean ",mean)

    variance=0

    for i in lis:
        deviation=i-mean    
        variance+=deviation**2

        print("deviation ",deviation)

    variance=variance/len(lis)
    print("\n\nvariance ",variance)

    standard_deviation=variance**(1/2)
    print("\n\nstandard_deviation ",standard_deviation)

    normalized_list=[]
    for i in lis:
        z=(i-mean)/standard_deviation
        normalized_list.append(z)


    return [normalized_list]
