from .utils import init_experiment, make_session


def create_recording(model, make_simulator, dir_name):
    simulator = make_simulator(record=True)
    state  = simulator.get_state()
    while state is not None:
        action = model.action(state)
        _ = simulator.take_action(action)
        state = simulator.get_state()

    simulator.save_recording(dir_name)

def record_mode(settings):
    session = make_session() # parallel session
    model, make_simulator = init_experiment(settings, session)
    create_recording(model, make_simulator, settings['__runtime__']['savedir'])
