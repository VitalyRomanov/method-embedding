class PythonCodeExamplesForNodes:
    examples = {
        "Assign": "a = 5\n",
        "AugAssign1": "a += 5\n",
        "AugAssign2": "a -= 5\n",
        "AugAssign3": "a *= 5\n",
        "AugAssign4": "a /= 5\n",
        "AugAssign5": "a //= 5\n",
        "AugAssign6": "a **= 5\n",
        "AugAssign7": "a |= True\n",
        "AugAssign8": "a &= True\n",
        "AugAssign9": "a >>= 5\n",
        "AugAssign10": "a <<= 5\n",
        "AugAssign11": "a %= 5\n",
        "AugAssign12": "a %= 5\n",
        "AugAssign13": "a @= 5\n",  # +=, -=, *=, /=, //=, **=, |=, &=, >>=, <<=, %= and ^=
        "Delete": "del a\n",
        "Global": "global av",
        "Nonlocal": "nonlocal a\n",
        "Slice": "a[0:5]\n",
        "ExtSlice": "a[0,1:5]\n",
        "Index": "a[0]\n",
        "Starred": "*a\n",
        "Yield": "yield a\n",
        "YieldFrom": "yield from a\n",
        "Compare1": "a == b\n",
        "Compare2": "a != b\n",
        "Compare3": "a > b\n",
        "Compare4": "a < b\n",
        "Compare5": "a >= b\n",
        "Compare6": "a <= b\n",
        "Compare7": "a in b\n",
        "Compare8": "a not in b\n",
        "Compare9": "a is b\n",
        "Compare10": "a is not b\n",
        "BinOp1": "a + b\n",
        "BinOp2": "a - b\n",
        "BinOp3": "a * b\n",
        "BinOp4": "a / b\n",
        "BinOp5": "a // b\n",
        "BinOp6": "a % b\n",
        "BinOp7": "a @ b\n",
        "BinOp8": "a & b\n",
        "BinOp9": "a ^ b\n",
        "BinOp10": "a ** b\n",
        "BinOp11": "a >> b\n",
        "BinOp12": "a >> b\n",
        "BoolOp1": "a and b\n",
        "BoolOp2": "a or b\n",
        "BoolOp3": "not a\n",
        "UnaryOp1": "+a\n",
        "UnaryOp2": "-a\n",
        "UnaryOp3": "~a\n",
        "Assert": "assert a == b\n",
        "FunctionDef":
            "def f(a):\n"
            "   return a\n",
        "AsyncFunctionDef":
            "async def f(a):\n"
            "   return a\n",
        "ClassDef":
            "class C:\n"
            "   def m():\n"
            "       pass\n",
        "AnnAssign": "a: int = 5\n",
        "With":
            "with open(a) as b:\n"
            "   do_stuff(b)\n",
        "AsyncWith":
            "async with open(a) as b:\n"
            "   do_stuff(b)\n",
        "arg":
            "def f(a: int = 5):\n"
            "   return a\n",
        "Await": "await call()\n",
        "Raise": "raise Exception()\n",
        "Lambda": "lambda x: x + 3\n",
        "IfExp": "a = 5 if True else 0\n",
        "keyword": "fn(a=5, b=4)\n",
        "Attribute": "a.b.c\n",
        "If":
            "if d is True:\n"
            "   a = b\n"
            "elif d is False:"
            "   b = a\n"
            "else:\n"
            "   a, a = c, c\n",
        "For":
            "for i in list:\n"
            "   k = fn(i)\n"
            "   if k == 4:\n"
            "       fn2(k)\n"
            "       break\n"
            "else:\n"
            "   fn2(0)\n",
        "AsyncFor":
            "async for i in list:\n"
            "   k = fn(i)\n"
            "   if k == 4:\n"
            "       fn2(k)\n"
            "       break\n"
            "else:\n"
            "   fn2(0)\n",
        "Try":
            "try:\n"
            "   a = b\n"
            "except Exception as e:\n"
            "   a = c\n"
            "else:\n"
            "   a = d\n"
            "finally:\n"
            "   print(a)\n",
        "While":
            "while b == c:\n"
            "   do_iter(b)\n",
        "Break":
            "while True:\n"
            "   break\n",
        "Continue":
            "while True:\n"
            "   continue\n",
        "Pass": "pass\n",
        "Dict": "{a:b, c:d}\n",
        "Set": "{a, c}\n",
        "ListComp": "[i for i in list]\n",
        "DictComp": "{i:j for i,j in list}\n",
        "SetComp": "{i for i in list}\n",
        "GeneratorExp": "(i for i in list if i != 5)\n",
        "BinOp": "c = a + b\n",
        "ImportFrom": "from module import Class\n",
        "alias": "import module as m\n",
        "List": "a = [1, 2, 3, 4]\n",
        "Tuple": "a = (1, 2, 3, 4)\n",
        "JoinedStr": "f'{a}'\n",
        "FormattedValue": "f'{an:.2f}'\n",
        "Bytes": "a = b'abc'\n",
        "Num": "a = 1\n",
        "Str": "a = 'abc'\n",
        "FunctionDef2": "@decorator\n"
                        "def function_name(\n"
                        "       arg0, /,\n"
                        "       arg1,\n"
                        "       arg2:int,\n"
                        "       \n"
                        "       key1:\n"
                        "            Dict[str, int] ='value1',\n"
                        "       key2=2,*,\n"
                        "       key3=None,\n"
                        "       key4=4,\n"
                        "       **kwargs\n"
                        ") -> returnedtype:\n"
                        "   pass\n",
        "FunctionDef3": "@decorator\n"
                        "def function_name(*args, **kwargs):\n"
                        "   pass\n"
    }
