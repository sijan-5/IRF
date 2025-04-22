import math
from math import comb, exp

#no important feature selected r
def compute_r(u, v, f):
    if v < f:
        return 0  # At least one important feature will beselected
    return comb(v, f) / comb(u + v, f)

def compute_q(u, v, f):
    r = compute_r(u, v, f)
    return 1 - r

def compute_strength(q, Nav, B):
    return 1 - (1 - q ** Nav) ** B

#correlation between two trees
def compute_correlation(u, v, f, Nav, B):
    total_features = u + v
    if total_features < 2 * f:
        return 1.0

    rho_prime = 1 - comb(total_features - f, f) / comb(total_features, f)
    rho = rho_prime ** Nav
    return 1 - (1 - rho) ** (B // 2)

u = 3
v = 2
f = 2
Nav = 3
B = 5

r = compute_r(u, v, f)
q = compute_q(u, v, f)
strength = round(compute_strength(q, Nav, B),4)
correlation = round(compute_correlation(u, v, f, Nav, B),4);

print(f"r (no important feature selected): {r}")
print(f"q (good split probability): {q}")
print(f"Strength (ηs): {strength}")
print(f"Correlation (ηc): {correlation}")