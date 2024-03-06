from main import MatrixEnsemble


def test_gen_matrix():
    a = MatrixEnsemble()
    result = a.generate_matrix(lambda: 1, 2)
    assert len(result) == 2
    for i in range(2):
        assert len(result[i]) == 2
        for j in range(2):
            assert result[i][j] == 1


def test_gen_ensemble():
    a = MatrixEnsemble(mode=1)
    for matrix in a.ensemble:

