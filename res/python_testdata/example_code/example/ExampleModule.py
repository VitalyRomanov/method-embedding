class ExampleClass:
    def __init__(self, argument: int):
        """
        Initialize. Инициализация
        :param argument:
        """
        self.field = argument

    def method1(self) -> str:
        """
        Call another method. Вызов другого метода.
        :return:
        """
        return self.method2()

    def method2(self) -> str:
        """
        Simple operations.
        Простые операции.
        :return:
        """
        variable1: int = self.field
        variable2: str = str(variable1)
        return variable2