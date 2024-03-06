import numpy as np


class MatrixEnsemble:
    @staticmethod
    def generate_matrix(gen_method: callable, size: int) -> np.ndarray:
        result = np.zeros([size, size])
        for i in range(size):
            for j in range(i, size):
                result[i][j] = gen_method()
                result[j][i] = result[i][j]
        return result

    def __init__(
        self,
        avg: float = 0.0,
        sigma: float = 1.0,
        size_ensemble: int = 1000,
        size_matrix: int = 2,
        mode: int = 0,
    ):
        self.norm_deltas = []
        self.size_ensemble = size_ensemble
        self.ensemble = []
        for i in range(size_ensemble):
            match mode:
                case 0:
                    self.ensemble.append(
                        self.generate_matrix(
                            lambda: np.random.normal(avg, sigma), size_matrix
                        )
                    )
                case 1:
                    self.ensemble.append(
                        self.generate_matrix(
                            lambda: np.random.choice([-1, 1]), size_matrix
                        )
                    )

    def gen_hist(self):
        for matrix in self.ensemble:
            eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))
            deltas = np.diff(eigenvalues)
            mean = np.mean(deltas)
            deltas_norm = deltas / mean
            for delta in deltas_norm:
                self.norm_deltas.append(delta)


if __name__ == "__main__":
    a = MatrixEnsemble()
    result = a.generate_matrix(lambda: 1, 2)
    assert len(result) == 2
    for i in range(2):
        assert len(result[i]) == 2
        for j in range(2):
            assert result[i][j] == 1
