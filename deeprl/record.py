from .utils import init_experiment, make_session


def create_recording(model, make_simulator, dir_name):
    simulator = make_simulator(record=True)
    state  = simulator.observe()
    while not simulator.is_terminal():
        action = model.action(state)
        _ = simulator.act(action)
        state = simulator.observe()

    simulator.execution_recording(dir_name)

def capture_metrics(model, make_simulator):
    simulator = make_simulator()
    state  = simulator.observe()
    while state is not None:
        action = model.action(state)
        _ = simulator.act(action)
        state = simulator.observe()

    return simulator.execution_metrics()

def record_mode(settings):
    session = make_session() # parallel session
    model, make_simulator = init_experiment(settings, session)
    create_recording(model, make_simulator, settings['__runtime__']['savedir'])
