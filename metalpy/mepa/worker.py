class Worker:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.group = name.split('-')[0]

        if self.group == name:
            print("WARNING: worker's name must follow the format '{group name}-{in-group id}', "
                  "and in-group id is assumed to start from zero,"
                  "if not, something may go wrong.")
            self.id = None
        else:
            self.id = int(name.split('-')[1])

    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight

    def get_group(self):
        return self.group

    def get_in_group_id(self):
        return self.id
