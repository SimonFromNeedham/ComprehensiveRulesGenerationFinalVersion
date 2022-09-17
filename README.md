# ComprehensiveRulesGenerationFinalVersion

Description: Exhaustively generates logical rules in Disjunctive Normal Form (DNF) from binary input. Categorial variables are one-hot-encoded and numerical variables are binarized with pre-set thresholds.

Performance:

2 GHz Dual-Core Intel Core i5 on 10,000 data points with 100 features each:
* 1 Feature combination: < 1 second
* 2 Feature combination: ~ 10 seconds
* 3 Feature combination: ~ 1 hour

Important Notes:
1. Uses the Pool.multiprocessing library for parallelism
2. DivideBy30 (Binary) and DivideBy30Remainder (Numerical) datasets are included for test purposes
3. For large datasets, the number of DNFs explored can be limited with the "NUM_FEATURES" constant at the top of main.py
