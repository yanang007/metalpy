class Worker:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        segments = name.split('-')
        # name中最后一个'-'前的内容 或 name，保证name不为空时，结果不为字符串
        self.group = '-'.join(segments[:-1]) or name

        if self.group == name or not segments[-1].isdigit():
            print("WARNING: worker's name must follow the format '{group name}-{in-group id}', "
                  "and in-group id is assumed to start from zero, "
                  "if not, something may go wrong.")
            self.id = None
        else:
            self.id = int(segments[-1])

    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight

    def get_group(self):
        return self.group

    def get_in_group_id(self):
        return self.id
