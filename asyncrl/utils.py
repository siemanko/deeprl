def import_class(path):
    path_split = path.split('.')
    # TODO(szymon): understand Python imports better

    # try to import entire module and then get class as an attribute
    try:
        module_name, class_name = '.'.join(path_split[:-1]), path_split[-1]
        module = __import__(module_name, fromlist=(class_name,))
        return getattr(module, class_name)
    except ImportError:
        pass

    # try to import only the main module and the getattr your way down.
    module_or_class = __import__(path_split[0], globals(), locals())
    for name in path_split[1:]:
        module_or_class = getattr(module_or_class, name)
    return module_or_class


def nps_to_bytes(arrays):
    memfile = BytesIO()
    memfile.write(('%d\n' % (len(arrays),)).encode('ascii'))
    for a in arrays:
        np.save(memfile, a)
    memfile.seek(0)
    b = memfile.read()
    return b

def bytes_to_nps(b):
    memfile = BytesIO(b)
    n_arrays = int(memfile.readline())
    return [np.load(memfile) for _ in range(n_arrays)]
