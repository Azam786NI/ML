from weight_init import initialize_weights
from forward_pass import calculate_forward_pass
from backpropagation import calculate_gradient
import numpy as np


# input_seq=[[1,2,3,4],[5,6,7,8]]
# output_seq=[[1,2],[5,6]]
input_seq=[1,2]
output_seq=[.1,.2]

weights,bias,ln_w,ln_b=initialize_weights(len(input_seq),len(output_seq),hidden_size=3,hidden_layers=2)

a,sa=calculate_forward_pass(input_seq,weights,bias,ln_w,ln_b)

loss=sa[-1]-output_seq
print("**********loss ",loss)

error=[]
for i in loss:
    error.append(i**2)

print("error ",0.5 * sum(error))

gradient=calculate_gradient(input_seq,a,sa,weights,loss)

for i in gradient:
    print(i,end="\n\n")

