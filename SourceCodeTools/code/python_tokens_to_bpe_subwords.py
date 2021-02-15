# "▁"
from functools import lru_cache

python_ops_to_bpe = {
    'Add': ["▁+"],
    'Mod': ["▁%"],
    'NotIn': ["▁not", "▁in"],
    'And': ["▁and"],
    'IsNot': ["▁is", "▁not"],
    'Or': ["▁or"],
    'Is': ["▁is"],
    'Eq': ["▁=="],
    'Gt': ["▁>"],
    'NotEq': ["▁!="],
    'Not': ["▁not"],
    'In': ["▁in"],
    'BitAnd': ["▁&"],
    'Sub': ["▁-"],
    'Mult': ["▁*"],
    'BitOr': ["▁|"],
    'Lt': ["▁<"],
    'Div': ["▁/"],
    'LtE': ["▁<="],
    'Pow': ["▁**"],
    'FloorDiv': ["▁//"],
    'GtE': ["▁>="],
    'USub': ["▁-"],
    'Invert': ["▁~"],
    'UAdd': ["▁+"],
    'MatMult': ["▁@"],
    'BitXor': ["▁^"],
    'LShift': ["▁<<"],
    'RShift': ["▁>>"],
    "Break": ["▁break"],
    "Pass": ["▁pass"],
    "Continue": ["▁continue"],
}

python_ops_to_literal = {
    'Add': "+",
    'Mod': "%",
    'NotIn': "not in",
    'And': "and",
    'IsNot': "is not",
    'Or': "or",
    'Is': "is",
    'Eq': "==",
    'Gt': ">",
    'NotEq': "!=",
    'Not': "not",
    'In': "in",
    'BitAnd': "&",
    'Sub': "-",
    'Mult': "*",
    'BitOr': "|",
    'Lt': "<",
    'Div': "/",
    'LtE': "<=",
    'Pow': "**",
    'FloorDiv': "//",
    'GtE': ">=",
    'USub': "-", # this appearsto be the operator to chenge number sign -5
    'Invert': "~",
    'UAdd': "+", # this appearsto be the operator to chenge number sign +5, method __pos__
    'MatMult': "@",
    'BitXor': "^",
    'LShift': "<<",
    'RShift': ">>",
    "Break": "break",
    "Pass": "pass",
    "Continue": "continue",
}

def op_tokenizer(op: str):
    return python_ops_to_bpe.get(op, op)

@lru_cache
def op_tokenize_or_none(op, tokenize_func):
    return tokenize_func(python_ops_to_literal[op]) if op in python_ops_to_literal else None