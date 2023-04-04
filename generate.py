#!/usr/bin/env python3

import sys
from sympy import symbols, expand
from sympy import cos, sin, tan, cosh, sinh, tanh, log, exp, sqrt, Id
from random import choice, random

MAX_TERMS = 5
CONSTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
x = symbols('x')
FUNS = [cos(x), sin(x), tan(x),
        cosh(x), sinh(x), tanh(x),
        log(x + 1), exp(x), sqrt(x + 1), Id(x)]

f = sin(x) * sin(x) + 2 * x + cos(x) + log(x + 1) + exp(x)

def taylor(f):
    return f.series(x, 0, 4).removeO()

def tokenize(f):
    f = str(f)
    f = f.replace('**', ' ^ ')
    f = f.replace('/', ' / ')
    f = f.replace('*', ' * ')
    f = f.replace('(', ' ( ')
    f = f.replace(')', ' ) ')
    f = f.replace('-', ' - ')
    f = f.replace('  ', ' ')
    f = f.strip()
    return f

def random_term():
    c = choice(CONSTS)
    f = choice(FUNS)
    sign = choice([1, -1])
    return sign * c * f

def random_small_function():
    n_terms = choice(range(1,MAX_TERMS))
    terms = [random_term() for _ in range(n_terms)]
    return sum(terms)

def random_function():
    c = choice(range(2))
    if c == 0:
        f = random_small_function()
    if c == 1:
        f0 = random_small_function()
        f1 = random_small_function()
        f2 = random_small_function()
        f = f0 * f1 + f2
    if c == 2:
        f0 = random_small_function()
        f1 = random_small_function()
        f2 = random_small_function()
        f3 = random_small_function()
        f4 = random_small_function()
        f = (f0 * f1 + f2) * f3 + f4
    return expand(f)

for _ in range(int(sys.argv[1])):
    f = random_function()
    t = taylor(f)
    print(f'{tokenize(f)}#{tokenize(t)}')


