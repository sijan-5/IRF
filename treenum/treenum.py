import numpy as np
import math

#η = λ (ηs - ηc)
def compute_accuracy(strength, correlation):
    lambda_val = 1
    return lambda_val * (strength - correlation)

# r = comb(v, f) / comb(u+v, f)
def compute_qu_qv(u, v, f):
    # Helper to compute r
    def compute_r(u, v, f):
        if v < f:
            return 0.0
        return math.comb(v, f) / math.comb(u + v, f)

    r_current = compute_r(u, v, f)
    r_u_plus = compute_r(u + 1, v, f)
    r_v_plus = compute_r(u, v + 1, f)
    qu = -(r_u_plus - r_current)  # approximate ∂q/∂u = -∂r/∂u
    qv = -(r_v_plus - r_current)  # approximate ∂q/∂v = -∂r/∂v
    return qu, qv

def compute_delta_u_v(u_old, u_new, v_old, v_new):
    delta_u = u_new - u_old
    delta_v = v_new - v_old
    return delta_u, delta_v

#ν
def compute_nu(q, rho, Nav, B):
    term1 = (1 - rho)**(B/2) / 2 * math.log(1 - rho) if rho < 1 else 0.0
    term2 = (1 - q**Nav)**B * math.log(1 - q**Nav) if q**Nav < 1 else 0.0
    return term1 - term2

#Compute l = B * Nav * q^(Nav-1) * (1 - q^Nav)^(B-1)
def compute_l(q, Nav, B):
    return B * Nav * q**(Nav - 1) * (1 - q**Nav)**(B - 1)

# Compute ΔB
def compute_deltaB(qu, qv, delta_u, delta_v, l, nu):
    numerator = l * (qu * delta_u + qv * delta_v)
    if nu == 0:
        return 0
    bound = abs(numerator / nu)
    return max(0, math.floor(bound))

# u_old, v_old = 3, 5  
# u_new, v_new = 4, 4  
# f = 3
# Nav = 10
# B = 20
# lambda_val = 1.0

# q = 0.8
# rho = 0.1
# strength = 1 - (1 - q**Nav)**B
# correlation = 1 - (1 - rho)**(B/2)

# qu, qv = compute_qu_qv(u_old, v_old, f)
# qu=round(qu,4)
# qv=round(qv,4)
# delta_u, delta_v = compute_delta_u_v(u_old, u_new, v_old, v_new)
# delta_u=round(delta_u,4)
# delta_v=round(delta_v,4)
# nu = round(compute_nu(q, rho, Nav, B),4)
# l = round(compute_l(q, Nav, B),4)
# deltaB = round(compute_deltaB(qu, qv, delta_u, delta_v, l, nu),4)
# accuracy = round(compute_accuracy(strength, correlation),4)

# print("qu:", qu)
# print("qv:", qv)
# print("Δu:", delta_u, "Δv:", delta_v)
# print("nu:", nu)
# print("l:", l)
# print("ΔB:", deltaB)
# print("Accuracy η:", accuracy)