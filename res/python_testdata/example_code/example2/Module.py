class Number:
    def __init__(self, value: int):
        """
        Initialize. Инициализация
        :param argument:
        """
        self.val = value

    def __add__(self, value):
        """
        Add two numbers.
        Сложить 2 числа
        :param value:
        :return:
        """
        return Number(self.val + value.val)

    def __repr__(self) -> str:
        """
        Return representation
        :return: Получить представление
        """
        return f"Number({self.val})"

