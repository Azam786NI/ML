from weight_init import initialize_weights
from forward_pass import calculate_forward_pass
from backpropagation import calculate_gradient
import numpy as np


input_seq = [1, 2]
output_seq = [0.1, 0.2]

weights, bias, ln_w, ln_b = initialize_weights(
    len(input_seq), len(output_seq), hidden_size=3, hidden_layers=2
)

z = calculate_forward_pass(input_seq, weights, bias, ln_w, ln_b)


loss = z[-1] - output_seq

error = []
for i in loss:
    error.append(i**2)

print("LOSS ", loss)
print("ERROR ", 0.5 * sum(error))

gradient = calculate_gradient(input_seq, z, weights, loss)

for i in gradient:
    print(i, end="\n\n")
