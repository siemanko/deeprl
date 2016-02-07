def import_class(path):
    path_split = path.split('.')
    module_name, class_name = '.'.join(path_split[:-1]), path_split[-1]
    module = __import__(module_name, fromlist=(class_name,))
    return getattr(module, class_name)
