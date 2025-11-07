# Determine bounds on the accuracy of a second classifer/rater (a2) given the
# accuracy of the first (a1) and Error Consisteny (k, Cohen's Kappa applied
# on correctness) between the two.


def upper_bound_on_accuracy_from_kappa(a1, k):
    a2_max = a1 * (2 - k) / (k + 2 * a1 * (1 - k))
    return a2_max


def lower_bound_on_accuracy_from_kappa(a1, k):
    a2_min = a1 * k / (2 * a1 * k - 2 * a1 - k + 2)
    return a2_min


def bounds_on_accuracy_from_kappa(a1, k):
    bounds = [
        lower_bound_on_accuracy_from_kappa(a1, k),
        upper_bound_on_accuracy_from_kappa(a1, k),
    ]
    return bounds


# bounds = bounds_on_accuracy_from_kappa(a1 = .9, k = .5)
# print(bounds) # [.75, .96]

# bounds = bounds_on_accuracy_from_kappa(a1 = .7, k = .5)
# print(bounds) # [.44, .87]

# bounds = bounds_on_accuracy_from_kappa(a1 = .5, k = .5)
# print(bounds) # [.25, .75]
