import tensorflow as tf
import os

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

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_experiment(settings, session):
    def make_simulator(record=False):
        simulator_class = import_class(settings['simulator']['class'])
        simulator       = simulator_class(settings['simulator']['settings'], record)
        return simulator

    model_class = import_class(settings['model']['class'])
    model       = model_class(settings['model']['settings'], session)

    return model, make_simulator

def make_session(max_cpu_cores=None):
    """Makes a multi-core session.
    If max_cpu_cores is None, it adopts the number of cores
    automatically
    """
    config = tf.ConfigProto()

    if max_cpu_cores is not None:
        config.device_count.update({'CPU': max_cpu_cores})

    return tf.Session(config=config)
