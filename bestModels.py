""" Generate and evaluate all possible DNF models given a subset size and a group of features """

from fastmetrics import fast_f1_score
from itertools import chain, combinations, permutations, product
from time import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from boolean import BooleanAlgebra
from numba import njit

global df
global df_test
global y_true
global subset_size
global num_models
global use_combinations

global y_true_np
global df_dict
global df_test_dict

global variables
global algebra


# Initialize global variables to be passed to each thread
def init_pools(df_temp, df_test_temp, y_true_temp, subset_size_temp, num_models_temp, use_combinations_temp):
    global df
    global df_test
    global y_true
    global subset_size
    global num_models
    global use_combinations

    global y_true_np
    global df_dict
    global df_test_dict

    global variables
    global algebra

    df = df_temp
    df_test = df_test_temp
    y_true = y_true_temp
    subset_size = subset_size_temp
    num_models = num_models_temp
    use_combinations = use_combinations_temp

    y_true_np = y_true.values
    df_dict = {}
    df_test_dict = {}

    for col in df.columns:
        df_dict[col] = df[col].values

    for col in df_test.columns:
        df_test_dict[col] = df_test[col].values

    variables = list(map(chr, range(122, 122 - subset_size, -1)))
    algebra = BooleanAlgebra()


# Simplify expression with boolean.py library
# In order to simplify and find sums in expression we need to
# replace 'df[columns[{i}]]' with one-character variables
def simplify_expr(expr, subset_size, variables, algebra):
    simple_expr = expr
    for i in range(subset_size):
        simple_expr = simple_expr.replace(f'df[columns[{i}]]', variables[i])
    simple_expr = str(algebra.parse(simple_expr).simplify())
    return simple_expr


@njit
def get_result(res, cols, result):
    for i in range(len(result)):
        if not result[i]:
            flag = True

            for j in range(len(res)):
                if res[j] != cols[i][j]:
                    flag = False
                    break

            result[i] = flag


# Code executed by each thread for evaluating template expressions
def best_model_helper(expr):
    simple_expr = simplify_expr(expr, subset_size, variables, algebra)

    # Replace one-character variables in simplified expr with 'df[columns[{i}]]'
    for i in range(subset_size):
        simple_expr = simple_expr.replace(variables[i], f"df[columns[{i}]]")

    # If the formula is a tautology
    if simple_expr == '1':
        return []

    min_f1 = -1
    best_formulas = []
    bool_sets = []

    # Split the DNF into "&" CNFs
    for cnf in expr.replace(" ", "").split('|'):
        bool_set = []

        # Record if each feature is true or false
        for part in cnf.split('&'):
            if part[0] == '~':
                bool_set.append(False)

            else:
                bool_set.append(True)

        bool_sets.append(np.array(bool_set))

    bool_sets = np.array(bool_sets)

    # Generate and evaluate all possible combinations or permutations depending on structure
    combos_or_perms = combinations(df.columns, len(bool_sets[0])) if use_combinations else \
        permutations(df.columns, len(bool_sets[0]))

    for columns in combos_or_perms:
        df_cols = []
        df_test_cols = []
        for col in columns:
            df_cols.append(df_dict[col])
            df_test_cols.append(df_test_dict[col])

        # .T to switch the rows and cols --> better cache access pattern
        df_cols = np.array(df_cols).T
        df_test_cols = np.array(df_test_cols).T

        result = np.full_like(y_true_np, False)
        test_result = [False] * len(df_test_cols)
        test_result = np.array(test_result)

        for bool_set in bool_sets:
            get_result(bool_set, df_cols, result)
            get_result(bool_set, df_test_cols, test_result)

        f1_score = fast_f1_score(y_true_np, result)

        if f1_score > min_f1:
            best_formulas.append((f1_score, columns, simple_expr, test_result, result))

        # Cut models to maintain memory
        if len(best_formulas) > 1.1*num_models:
            best_formulas.sort(reverse=True)
            best_formulas = best_formulas[:num_models]
            min_f1 = best_formulas[-1][0]

    best_formulas.sort(reverse=True)
    best_formulas = best_formulas[:num_models]
    return best_formulas


# Generator of boolean formulas in str format using truth tables
def model_string_gen(vars_num):
    inputs = list(product([False, True], repeat=vars_num))
    for output in product([False, True], repeat=len(inputs)):
        terms = []

        for j in range(len(output)):
            if output[j]:
                terms.append(' & '.join(['df[columns[' + str(i) +']]' if input_
                                         else '~df[columns[' + str(i) +']]' for i, input_ in enumerate(inputs[j])]))

        if not terms:
            continue

        expr = ' | '.join(terms)
        yield expr


# Driver code for generating, evaluating, and compiling best models
def find_best_model(df, df_test, y_true, subset_size, parallel, num_threads, num_models, use_combinations):
    formula_cnt = max(2 ** (2 ** subset_size) - 1, 10)
    best_formulas = []

    if subset_size == 1:
        init_pools(df, df_test, y_true, subset_size, num_models, use_combinations)

        start = time()
        for expr in model_string_gen(subset_size):
            best_formulas += best_model_helper(expr)

    elif parallel:
        # Implement parallelism using pooling
        pool = Pool(processes=num_threads, initializer=init_pools, initargs=(df, df_test, y_true, subset_size,
                                                                             num_models, use_combinations))

        start = time()
        best_formulas = [pool.map(best_model_helper, tqdm(model_string_gen(subset_size), total=formula_cnt))]

        # Join together the best models of each thread
        best_formulas = list(chain.from_iterable(best_formulas))
        best_formulas = list(chain.from_iterable(best_formulas))

    else:
        init_pools(df, df_test, y_true, subset_size, num_models, use_combinations)

        start = time()
        for expr in tqdm(model_string_gen(subset_size), total=formula_cnt):
            best_formulas += best_model_helper(expr)

    elapsed_time = time() - start
    print("Elapsed time:", round(elapsed_time, 3), "seconds")
    return best_formulas
