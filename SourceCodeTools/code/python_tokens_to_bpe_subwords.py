# "▁"
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
    # 'USub':
    'Invert': ["▁~"],
    # 'UAdd'
    # 'MatMul??'
    'BitXor': ["▁^"],
    'LShift': ["▁<<"],
    'RShift': ["▁>>"],
    "Break": ["▁break"],
    "Pass": ["▁pass"],
    "Continue": ["▁continue"],
}

def op_tokenizer(op: str):
    return python_ops_to_bpe.get(op, op)