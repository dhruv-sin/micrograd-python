import numpy as np

P_id = np.array([0, 1, 2])
A_R  = np.array([0, 1, 2])
B_T  = np.array([4, 5, 6])
n = len(B_T)

L = np.tril(np.ones((n, n)))
C_T = np.einsum('ij,j->i', L, B_T)

TAT = C_T - A_R


W_T = TAT - B_T


R_T = (C_T - B_T) - A_R

final_table = np.column_stack((P_id, A_R, B_T, C_T, TAT, W_T, R_T))

print("P_id  Arr  Bur  Com  TAT  W_T  R_T")
print(final_table)
