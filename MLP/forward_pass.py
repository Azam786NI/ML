import numpy as np
from Utils.LayerNorm import LN

def handel_activation(cal,activation):

    new_cal=[]

    if activation=='ReLU':
        for i in cal[0]:
            if i<=0:
                new_cal.append(0)
            else:
                new_cal.append(i)
        return np.array(new_cal)
    
    elif activation=="sigmoid":
        return 1/(1+np.exp(-cal))

    elif activation=="tanh":
        return np.tanh(cal)
    
    else:
        print("no activation applied")
        return cal

def calculate_forward_pass(input_seq,weights,bias,ln_w,ln_b,activation="ReLU"):
    a=[]
    sa=[]
    last_step=input_seq

    idx=0
    for w,b in zip(weights,bias): 
        idx+=1

        # weight * sa * bias
        cal=np.dot(last_step,w)
        cal+=b
        a.append(cal)
        print(f"------------- Z{idx}",cal)

        # LN
        if idx!=len(w):
            cal=LN(cal)
            print(f"------------- Z_norm {cal}")

            cal=(cal * ln_w[idx-1]) + ln_b[idx-1]
            print(f"------------- Z_scale {cal}")

            # activation         
            cal=handel_activation(cal,activation)
        
        sa.append(cal)
        print(f"------------- A{idx}",cal,end=f"\n\n=== pass{idx} completed ====\n\n")

        last_step=cal

    return a,sa

