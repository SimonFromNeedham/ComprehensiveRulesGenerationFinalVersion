""" Beautifies a formula by...
        a. Combining columns and expression
        b. Adding True Positive, False Positive, etc.
        c. Putting the data into an Excel readable format """

import fastmetrics
from numba import vectorize
import numpy as np
from itertools import combinations


@vectorize
def generate_confusion_matrix(x, y):
    """
    NumPy ufunc implemented with Numba that generates a confusion matrix as follows:
    1 = True Positive, 2 = False Positive, 3 = False Negative, 4 = True Negative.
    """
    if x and y:
        return 1

    elif not x and y:
        return 2

    elif x and not y:
        return 3

    else:
        return 4


def count_confusion_matrix(y_true, y_pred):
    matrix = generate_confusion_matrix(y_true, y_pred)
    tp = np.count_nonzero(matrix == 1)  # True Positive
    fp = np.count_nonzero(matrix == 2)  # False Positive
    fn = np.count_nonzero(matrix == 3)  # False Negative
    tn = np.count_nonzero(matrix == 4)  # True Negative

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    tnr = tn / (tn+fp)
    fnr = fn / (fn+tp)

    return tpr, fpr, tnr, fnr


# Generate metrics the user might want to sort by
def generate_metrics(y_true, y_pred):
    precision = fastmetrics.fast_precision_score(y_true, y_pred)
    recall = fastmetrics.fast_recall_score(y_true, y_pred)
    roc = fastmetrics.fast_roc_auc_score(y_true, y_pred)
    acc = fastmetrics.fast_accuracy_score(y_true, y_pred)
    return precision, recall, roc, acc


# A bunch of functions to find "summation" forms for potential formulas
def sums_generator(subset_size):
    variables = list(map(chr, range(122, 122 - subset_size, -1)))
    indices = list(range(0, subset_size, 1))

    sum_dict = {}
    for i in range(1, subset_size + 1):
        sum_dict[i] = []
        sum_dict[i].append(list(map(set, list(combinations(variables, i)))))
        for j in range(1, subset_size + 1):
            indices_comb = list(combinations(indices, j))
            for inds in indices_comb:
                variables_with_tilda = variables.copy()
                for ind in inds:
                    variables_with_tilda[ind] = '~' + variables[ind]
                sum_dict[i].append(list(map(set, list(combinations(variables_with_tilda, i)))))

    return sum_dict


def formula_partition(s):
    s = s.replace('(','').replace(')','')
    list_of_exp = s.split('|')
    sum_list = []
    for sub_expr in list_of_exp:
        if '&' not in sub_expr:
            sum_list.append(set([sub_expr]))
        else:
            sum_list.append(set(sub_expr.split('&')))
    return sum_list


def find_one_sum(sum_dict, f):
    sum_list = formula_partition(f)
    for sum_key in sum_dict.keys():
        for sub_sum in sum_dict[sum_key]:
            c = 0
            len_sum = len(sub_sum)
            parts_to_replace = []
            set_of_vars = set()
            for sub_exp in sub_sum:
                if sub_exp in sum_list:
                    c += 1
                    parts_to_replace.append(sub_exp)
                    for var in sub_exp:
                        set_of_vars.add(var)
            if c == len_sum:
                if len(parts_to_replace) < len(sum_list):
                    for part in parts_to_replace:
                        sum_list.remove(part)
                    rest_of_formula = str(sum_list)[2:-2].replace('\'', '').replace('}, {', '|').replace(', ', '&')
                    sum_formula = 'sum({})>={}'.format(set_of_vars, sum_key)
                    # formula = 'sum({})>={}|{}'.format(set_of_vars, sum_key, rest_of_formula)
                else:
                    # formula = 'sum({})>={}'.format(set_of_vars, sum_key)
                    rest_of_formula = None
                    sum_formula = 'sum({})>={}'.format(set_of_vars, sum_key)
                return sum_formula, rest_of_formula


def find_sum(sum_dict, f):
    rest_of_formula = f
    sums = []

    while rest_of_formula is not None and find_one_sum(sum_dict, rest_of_formula) is not None:
        sum_formula, rest_of_formula = find_one_sum(sum_dict, rest_of_formula)
        sums.append(sum_formula)

    if len(sums) > 0:
        if rest_of_formula is not None:
            sums.append(rest_of_formula)
        return '|'.join(c for c in sums)

    return ""


def simplify(cols, expr, subset_size):
    # Separate DNF and CNF components
    expr = expr.replace("&", "  &  ")
    expr = expr.replace("|", "  |  ")

    for i in range(subset_size):
        expr = expr.replace(f"df[columns[{i}]]", cols[i])

    return expr


# Returns: [TPR, FPR, FNR, TNR, F1 Score, Simple DNF, Summed DNF]
def beautify_formulas(best_formulas, y_true, subset_size):
    results = []
    sums = sums_generator(subset_size)

    for formula in best_formulas:
        res = [fastmetrics.fast_f1_score(y_true, formula[-2])]

        # Generate true positive rate, false positive rate, etc.
        for val in count_confusion_matrix(y_true, formula[-2]):
            res.append(val)

        for val in generate_metrics(y_true, formula[-2]):
            res.append(val)

        # Create a fluid final expression
        cols = formula[1]
        simple_expr = formula[2]
        summed_expr = find_sum(sums, simple_expr)

        simple_expr = simplify(cols, simple_expr, subset_size)
        summed_expr = simplify(cols, summed_expr, subset_size)

        res.append(simple_expr)
        res.append(summed_expr)
        results.append(res)

    return sorted(results, reverse=True)
