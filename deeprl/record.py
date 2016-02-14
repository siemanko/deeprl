from .utils import init_experiment, make_session


def create_recording(model, simulator, dir_name):
    reward = 0.0
    while reward is not None:
        state  = simulator.get_state()
        action = model.action(state)
        reward = simulator.take_action(action)

    simulator.save_recording(dir_name)

def record_mode(settings):
    session = make_session() # parallel session
    model, simulator = init_experiment(settings, session, record=True)
    create_recording(model, simulator, settings['__runtime__']['savedir'])
