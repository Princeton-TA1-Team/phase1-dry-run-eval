"""Evaluator for the 24-game task.

The verifier matches answers of the form `(a op b) op (c op d) [= 24]`
against a target int by re-evaluating the arithmetic expression and
checking that the digits used are a permutation of the input set.
"""
