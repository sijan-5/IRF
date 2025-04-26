from math import comb

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
    total = u + v
    if total < 2 * f:
        # no matter what, return (corr, rho)
        rho = 1.0
        corr = 1 - (1 - rho)**(B//2)   # which is also 1.0
        return corr, rho

    rho_prime = 1 - comb(total - f, f) / comb(total, f)
    rho = rho_prime ** Nav
    corr = 1 - (1 - rho) ** (B // 2)
    return corr, rho


# u = 3
# v = 2
# f = 2
# Nav = 3
# B = 5

# r = compute_r(u, v, f)
# q = compute_q(u, v, f)
# strength = round(compute_strength(q, Nav, B),4)
# correlation, rho = compute_correlation(u, v, f, Nav, B)
# correlation=round(correlation,4)
# print(f"r (no important feature selected): {r}")
# print(f"q (good split probability): {q}")
# print(f"Strength (ηs): {strength}")
# print(f"Correlation (ηc): {correlation}")