import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Use the precision you already set
mp.dps = 100

# ----- Your functions (F, R, R_inv, psi) go here -----
def F(z, d):
    """
    Apply one iteration of the operator F to a probability distribution z.

    This function computes the distribution at the root of a d‑ary tree of height n
    given the distribution at the root of its subtrees of height n‑1 (represented by z).
    The operation corresponds to the recursion:

        A(z)_i = (z_{i-1} + z_i + z_{i+1})^d,      F(z) = A(z) / sum(A(z))

    Parameters
    ----------
    z : list of mpf numbers
        A list of length n representing a probability distribution over consecutive
        integer values. Index 0 corresponds to the smallest integer in the support,
        index n-1 to the largest. The values must be non‑negative and sum to 1.
    d : int
        The degree of the tree (branching factor).

    Returns
    -------
    list of mpf numbers
        A new list of length n+2 containing the next‑level distribution.
        Its support is extended by one on each side relative to z.
        The entries are normalized so that they sum to 1.

    Notes
    -----
    - The function assumes that outside the range [0, n-1] the probability is zero.
      Hence the boundary indices of the new distribution are computed using only
      the available neighbours from z.
    - High precision arithmetic (mpmath) is used throughout, which is essential
      for the very small probabilities that appear when d is large or after many
      iterations.
    - The input list does not need to be symmetric, but the context of the paper
      often deals with symmetric distributions around zero.
    - After applying F, the output list has length increased by 2. When iterating,
      remember to keep track of the integer coordinate corresponding to each index.
    """
    n = len(z)
    z_new = [mp.mpf(0)] * (n + 2)

    # Leftmost new index (original leftmost - 1): only neighbour at original leftmost contributes
    z_new[0] = z[0] ** d
    # Next index (original leftmost): neighbours at original leftmost and leftmost+1
    z_new[1] = (z[0] + z[1]) ** d

    # Rightmost new index (original rightmost + 1)
    z_new[-1] = z[-1] ** d
    # Previous index (original rightmost)
    z_new[-2] = (z[-1] + z[-2]) ** d

    # Interior indices: three neighbours available
    for i in range(2, n + 2 - 2):   # i runs from 2 to n-1 (inclusive) in the new list
        # z[i-2], z[i-1], z[i] correspond to the three neighbours in the original list
        z_new[i] = (z[i-2] + z[i-1] + z[i]) ** d

    # Normalize to obtain a probability distribution
    total = mp.fsum(z_new)
    z_new = [x / total for x in z_new]
    return z_new




def R(z):
    """
    Compute the ratio map R(z) for a symmetric probability distribution z.
    
    Parameters
    ----------
    z : list of mpf numbers
        A symmetric list of values around the centre, e.g. 
        z[0] = value at -L, z[L] = value at 0, z[-1] = value at +L.
        (Assumed length is odd.)
    
    Returns
    -------
    list of mpf numbers
        Ratios x_i = z_i / z_{i-1} for i = 1, 2, ..., L, where L = (len(z)-1)//2.
        If z_{i-1} == 0, x_i is set to 0.
    """
    L = (len(z) - 1) // 2          # maximum distance from the centre
    centre = L                     # index of the centre (distance 0)
    
    ratios = []
    for i in range(1, L+1):        # i = distance
        num   = z[centre + i]      # value at distance i
        denom = z[centre + i - 1]  # value at distance i-1
        if denom == 0:
            ratios.append(mp.mpf(0))
        else:
            ratios.append(num / denom)
    return ratios




def R_inv(x):
    """
    Compute the inverse of the ratio map R.

    Given a list x = [x_1, x_2, ..., x_L] of ratios, this function returns the
    symmetric probability distribution z (centered at zero) such that R(z) = x.

    The construction follows the explicit formula derived from the definition
    of R:  x_i = z_i / z_{i-1}  for i ≥ 1, with z_0 as the value at distance 0.
    For a symmetric distribution we have z_{-i} = z_i. The probabilities are
    normalised so that z_0 + 2·∑_{i=1}^{L} z_i = 1.

    Parameters
    ----------
    x : list of mpf numbers
        A list of length L representing the ratios x_1, x_2, … , x_L.
        All entries must be non‑negative.

    Returns
    -------
    list of mpf numbers
        A list of length 2L+1 representing the symmetric probability
        distribution z. The centre (distance 0) is at index L.
        The entries are normalised so that they sum to 1.

    Notes
    -----
    - The computation uses mpmath high‑precision arithmetic.
    - If any ratio x_j = 0, all subsequent products become zero,
      which is handled correctly.
    - The input is assumed to belong to the image of R (i.e., it satisfies
      the necessary convergence conditions), but the function will still
      produce a normalised list even if the sum of products diverges (in
      that case the denominator might be infinite, which would make z₀ = 0).
    - The length L of the input determines how many positive‑side values
      are returned; further ratios (beyond L) are assumed to be zero.
    """
    L = len(x)
    # Compute prefix products p[i] = prod_{j=1}^{i} x_j
    p = [mp.mpf(1)] * L       # p[i] corresponds to product up to i (1‑based)
    current = mp.mpf(1)
    for i in range(L):
        current *= x[i]
        p[i] = current

    # Denominator for z_0 = 1 / (1 + 2·∑ p[i])
    sum_prod = mp.fsum(p)
    denominator = 1 + 2 * sum_prod
    z0 = 1 / denominator

    # Build the symmetric list of length 2L+1
    total_len = 2 * L + 1
    z = [mp.mpf(0)] * total_len
    centre = L
    z[centre] = z0

    for i in range(1, L + 1):
        val = p[i-1] * z0
        z[centre + i] = val
        z[centre - i] = val

    return z




def psi(x, d):
    """
    Apply one iteration of the operator ψ to a ratio sequence x.

    ψ is defined as ψ = R ∘ F ∘ R⁻¹, where R maps a symmetric probability
    distribution to its ratios x_i = z_i / z_{i-1} (i ≥ 1).  The explicit
    formulas derived from the definition are:

        ψ_1(x) = ((1 + x_1 + x_1 x_2) / (1 + 2 x_1))^d
        ψ_n(x) = x_{n-1}^d * ((1 + x_n + x_n x_{n+1}) / (1 + x_{n-1} + x_{n-1} x_n))^d   (n ≥ 2)

    These formulas involve x_{n-1} even for the largest index, so the length
    of the non‑zero part of the sequence can increase.  Starting from a finite
    list x = [x_1, ..., x_L] (with x_{L+1} = x_{L+2} = ... = 0), the image ψ(x)
    will generally have non‑zero entries up to index L+1.  Indeed,
    ψ_{L+1}(x) = (x_L / (1 + x_L))^d, which is positive whenever x_L > 0.
    Hence the output list has length L+1.

    Parameters
    ----------
    x : list of mpf numbers
        A list representing the ratios x_1, x_2, ..., x_L.  All entries must be
        non‑negative.  The length L can be 0 (empty list), in which case the
        input is interpreted as the infinite zero sequence.
    d : int
        The degree of the tree (branching factor).

    Returns
    -------
    list of mpf numbers
        A list y of length L+1 (or length 1 if L=0) containing the ratios
        ψ_1(x), ψ_2(x), …, ψ_{L+1}(x).  All entries are non‑negative.

    Notes
    -----
    - The formulas are evaluated using mpmath high‑precision arithmetic.
    - When L=0 (empty list), the input corresponds to all ratios zero, and
      the output is [1] because ψ_1(0) = 1.
    - The output length is always one greater than the input length because
      the recurrence for ψ_{L+1} uses only x_L (and the implicitly zero
      x_{L+1}, x_{L+2}) and can be non‑zero.  This expansion is essential
      for correctly iterating ψ and matches the theoretical behaviour of the
      infinite‑dimensional map.
    """
    L = len(x)

    # Special case: empty list -> all ratios are zero
    if L == 0:
        return [mp.mpf(1)]

    # Output list of length L+1 (indices 0 … L)
    y = [mp.mpf(0)] * (L + 1)

    # ψ_1(x) = ((1 + x_1 + x_1 x_2) / (1 + 2 x_1))^d
    x1 = x[0]
    x2 = x[1] if L >= 2 else mp.mpf(0)
    numerator1 = 1 + x1 + x1 * x2
    denominator1 = 1 + 2 * x1
    y[0] = (numerator1 / denominator1) ** d

    # ψ_n for n = 2 … L
    for n in range(2, L + 1):          # n is the index in the infinite sequence
        idx = n - 1                     # corresponding position in y
        x_prev = x[n - 2]               # x_{n-1}
        x_curr = x[n - 1]               # x_n
        # x_{n+1} is zero if n == L, otherwise x[n]
        x_next = x[n] if n <= L - 1 else mp.mpf(0)

        numerator_n = 1 + x_curr + x_curr * x_next
        denominator_n = 1 + x_prev + x_prev * x_curr
        y[idx] = (x_prev ** d) * ((numerator_n / denominator_n) ** d)

    # ψ_{L+1} – only requires x_L (the last element of x)
    # Formula: ψ_{L+1} = x_L^d * ((1 + 0 + 0) / (1 + x_L + x_L·0))^d = (x_L / (1 + x_L))^d
    if L >= 1:
        x_last = x[-1]
        y[L] = (x_last / (1 + x_last)) ** d

    return y





# ----- Parameters -----
L_half =4                     # half‑length: support from -L_half to L_half
epsilon = mp.mpf('1e-10')       # small probability on interior points
d = 2                          # branching factor (change as needed)
y_max = 10**100                # cutoff for values that get too big 

# ----- Build initial symmetric distribution z0 -----
# Support indices: -10, -9, ..., 9, 10  → 21 points
z0 = [mp.mpf(0)] * (2 * L_half + 1)

# # Probability at the two ends: make total sum = 1
# a = (1 - (2 * L_half - 1) * epsilon) / 2   # ~0.5
# z0[0] = a                # -10
# z0[-1] = a               # +10
# for i in range(1, 2 * L_half):   # interior indices -9 … 9
#     z0[i] = epsilon
z0[L_half+1] = 1

# ----- Compute initial ratios x0 = R(z0) -----
x = R(z0)             # list of length 10 (ratios x_1, …, x_10)
print("Initial ratios (x0):", [float(v) for v in x])


import math
import matplotlib.animation as animation

# ----- Compute iterations of psi -----
num_iterations = 1000           # number of frames (change as desired)
x_seq = [x]                   # start with the initial ratios from above
for _ in range(num_iterations):
    x_seq.append(psi(x_seq[-1], d))


# ----- Compute the corresponding probability distributions via R_inv -----
z_seq = [R_inv(x) for x in x_seq]   # each z is a symmetric list of length 2*len(x)+1


# Probabilities plot
max_len_probs = max(len(zi) for zi in z_seq)          # total length, odd
half_len = (max_len_probs - 1) // 2
x_min_probs, x_max_probs = -half_len - 0.5, half_len + 0.5   # leave margin


# ----- Set up the figure with two subplots -----
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 5))

# Left subplot: ratios
ax1.set_xlim(0, 30)
ax1.set_ylim(-1, 30)
ax1.set_xlabel('Index i')
ax1.set_ylabel('Ratio $x_i$')
ax1.set_title('Iteration 0')

# Right subplot: probabilities
ax2.set_xlim(-L_half-5, L_half+5)
ax2.set_ylim(-0.025, 1)
ax2.set_xlabel('Distance from centre')
ax2.set_ylabel('Probability')
ax2.set_title('Distribution $F^n(z_0)$')

# Create the line objects (initially empty)
line_ratios, = ax1.plot([], [], 'b-o', markersize=4, label='ratios')
line_probs,  = ax2.plot([], [], 'r-o', markersize=4, label='probabilities')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

# ----- Animation functions -----
def init():
    line_ratios.set_data([], [])
    line_probs.set_data([], [])
    return line_ratios, line_probs

def update(frame):
    # Update ratios (left subplot)
    x_vals = x_seq[frame]
    indices_ratios = list(range(1, len(x_vals) + 1)) + [len(x_vals) + 1]
    values_ratios = []
    for v in x_vals:
        try:
            fv = float(v)
            if not math.isfinite(fv):
                fv = y_max
        except (OverflowError, ValueError):
            fv = y_max
        values_ratios.append(fv)
    values_ratios.append(0.0)   # extra zero point
    line_ratios.set_data(indices_ratios, values_ratios)

    # Update probabilities (right subplot)
    z_vals = z_seq[frame]               # symmetric list, centre at index L
    Lz = len(z_vals)
    centre = (Lz - 1) // 2
    # x‑coordinates: distances from centre
    indices_probs = list(range(-centre, centre + 1))
    values_probs = []
    for v in z_vals:
        try:
            fv = float(v)
            if not math.isfinite(fv):
                fv = y_max
        except (OverflowError, ValueError):
            fv = y_max
        values_probs.append(fv)
    line_probs.set_data(indices_probs, values_probs)

    # Update titles
    ax1.set_title(f'Iteration {frame} (ratios)')
    ax2.set_title(f'Iteration {frame} (distribution)')
    return line_ratios, line_probs

# ----- Create and show the animation -----
anim = animation.FuncAnimation(
    fig, update, frames=len(x_seq),
    init_func=init, blit=False, repeat=True
)

plt.tight_layout()
plt.show()