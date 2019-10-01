import importlib


def make(config_data, section_name):
    attr = get_attr(config_data, section_name)
    return attr(config_data, section_name)


def get_attr(config_data, section_name):
    name = config_data.get(section_name, 'name')
    module = config_data.get(section_name, 'module')
    module = importlib.import_module(module)
    attr = getattr(module, name)
    return attr
