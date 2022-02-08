import ast


def has_valid_syntax(function_body):
    try:
        ast.parse(function_body.lstrip())
        return True
    except SyntaxError:
        return False
