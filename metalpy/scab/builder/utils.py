class MissingArgException(ValueError):
    def __init__(self, arg_name: str, arg_sources: list[str]):
        super().__init__(arg_name, arg_sources)

    @property
    def arg_name(self):
        return self.args[0]

    @property
    def arg_sources(self):
        return self.args[1]

    def __str__(self):
        arg_sources = [f'`{s}`' for s in self.arg_sources]
        ni = max(len(arg_sources) - 1, 1)
        suppliers = " or ".join([", ".join(arg_sources[:ni]), *arg_sources[ni:]])
        return f'Missing required arg `{self.arg_name}`' \
               f' specify it with {suppliers}'


class MissingArgsException(Exception):
    def __init__(self):
        super().__init__([])

    @property
    def _missing_args(self) -> list[MissingArgException]:
        return self.args[0]

    def append(self, ex: MissingArgException):
        self._missing_args.append(ex)

    def __str__(self):
        return '\n'.join((str(arg) for arg in self._missing_args))
