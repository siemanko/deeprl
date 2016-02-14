from multiprocessing import (
    Process,
    Queue as ProcessQueue,
)

from .utils import init_experiment, make_session


def parameter_server_process(savedir, settings, to_ps, from_ps):
    session = make_session(max_cpu_cores=1)
    model, _ = init_experiment(settings, session, record=True)

    while True:
        msg = to_ps.get()
        print(msg)

def worker_process(settings, to_pos, from_ps):
    while True:
        print("Sending...")
        to_pos.put('siema!')
        time.sleep(2)



def parallel_mode(settings):
    num_workers = settings["num_workers"]
    print ("Initializing %d workers" % (num_workers,))



    to_ps = ProcessQueue()
    from_ps = [ProcessQueue() for _ in range(num_workers)]

    ps = Process(target=parameter_server_process, args=(savedir, settings, to_ps, from_ps))

    workers = []
    for i in range(num_workers):
        workers.append(Process(target=worker_process, args=(settings, to_ps, from_ps[i])))

    ps.start()
    for worker in workers:
        worker.start()
