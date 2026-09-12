"""Small adapters around MP-SPDZ's existing layers, SGD and persistence."""
from Compiler import ml
from Compiler.library import for_range_opt, start_timer, stop_timer
from Compiler.types import cfix, sfix


class RegularizedDense(ml.Dense):
    def reset(self):
        # Fixed zero initialization makes the clear reference reproducible.
        self.W.assign_all(0)
        self.b.assign_all(0)

    def backward_params(self, derivative, batch):
        super().backward_params(derivative, batch)
        # Upstream SGD divides by the batch length; regularize W, never b.
        self.nabla_W.assign_vector(self.nabla_W.get_vector()
                                   + (len(batch) * 0.01) * self.W.get_vector())


class HingeOutput(ml.Output):
    """Mean hinge loss gradient; all margin bits remain secret."""
    def __init__(self, n):
        super().__init__(n, approx=False)
        self.compute_loss = False

    def _forward(self, batch):
        self.l.write(0)

    def backward(self, batch):
        n = len(batch)
        y = self.Y.get(batch.get_vector(0, n))
        z = self.X.get_vector(0, n)
        active = y * z < 1
        self.nabla_X.assign_vector(active.if_else(-y, 0))

    def eval(self, size, base=0, top=False):
        value = self.X.get_vector(base, size)
        return (value > 0) if top else value


def run_secure_ml(model_name, phase):
    n, d = map(int, program.args[1:3])
    epochs = int(program.args[3]) if phase == 'train' else 0
    assert d == 4 and n > 0
    sfix.set_precision(16, 31)
    cfix.set_precision(16, 31)
    sfix.round_nearest = True
    ml.Layer.back_batch_size = n
    ml.set_n_threads(1)
    dense = RegularizedDense(n, d, 1)
    output = ml.Output(n, approx=5) if model_name == 'lr' else HingeOutput(n)
    output.compute_loss = False

    start_timer(1)
    halves = [sfix.Matrix(n, d // 2) for _ in range(2)]
    for party in range(2):
        halves[party].input_from(party)
    @for_range_opt(n)
    def load_row(i):
        for j in range(d):
            dense.X[i][0][j] = halves[j // (d // 2)][i][j % (d // 2)]
    if phase == 'train':
        output.Y.input_from(0)
    else:
        _, weights = sfix.read_from_file(0, n_items=d + 1, crash_if_missing=True)
        for j in range(d):
            dense.W[j][0] = weights[j]
        dense.b[0] = weights[d]
    stop_timer(1)

    start_timer(2)
    if phase == 'train':
        graph = ml.SGD([dense, output], n_epochs=epochs, report_loss=False)
        graph.momentum = 0.0
        graph.shuffle = False
        graph.revealing_correctness = False
        graph.set_learning_rate(0.5)
        graph.reset()
        graph.run(batch_size=n)
    else:
        # Inference does not construct an SGD optimizer, reset or update weights.
        graph = ml.Optimizer([dense, output], report_loss=False)
    graph.forward(N=n, run_last=False)
    scores = output.eval(n)
    classes = scores > (0.5 if model_name == 'lr' else 0)
    stop_timer(2)

    start_timer(3)
    if phase == 'train':
        sfix.write_to_file([dense.W[j][0] for j in range(d)] + [dense.b[0]], position=0)
    stop_timer(3)

    # Public-data diagnostics only. Inference reloads shares, not this output.
    start_timer(4)
    if phase == 'train':
        dense.W.get_vector().v.reveal_to(0).binary_output()
        dense.b.get_vector().v.reveal_to(0).binary_output()
    scores.v.reveal_to(0).binary_output()
    classes.reveal_to(0).binary_output()
    stop_timer(4)
