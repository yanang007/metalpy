
def structured_traverse(struct, func):
    if isinstance(struct, dict):
        ret = {key: structured_traverse(element, func) for key, element in struct}
    elif isinstance(struct, list):
        ret = [structured_traverse(element, func) for element in struct]
    else:
        ret = func(struct)

    return ret
