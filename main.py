import numpy as np
import matplotlib.pyplot as plt


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
        matrix_sizes: tuple[int] = (2, 4, 16),
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
                            lambda: np.random.normal(avg, sigma),
                            np.random.choice(matrix_sizes),
                        )
                    )
                case 1:
                    self.ensemble.append(
                        self.generate_matrix(
                            lambda: np.random.choice([-1, 1]),
                            np.random.choice(matrix_sizes),
                        )
                    )

    def gen_hist(self):
        for matrix in self.ensemble:
            eigenvalues = np.real(np.linalg.eigvals(matrix))
            sorted_eigenvalues = np.sort(eigenvalues)
            diffs = np.diff(sorted_eigenvalues)
            mean_diff = np.mean(diffs)
            normalized_diffs = diffs / mean_diff
            for diff in normalized_diffs:
                self.norm_deltas.append(diff)
        return self.norm_deltas


if __name__ == "__main__":
    a = MatrixEnsemble(mode=1)
    deltas = a.gen_hist()
    x = np.sort(np.array(deltas))
    y = np.pi * x / 2 * np.power(np.e, -np.pi * x**2 / 4)
    plt.figure()
    plt.hist(
        deltas,
        bins=20,
        density=True,
    )
    plt.plot(x, y, color="r")
    plt.xlabel("Значение")
    plt.ylabel("Количество")
    plt.show()
