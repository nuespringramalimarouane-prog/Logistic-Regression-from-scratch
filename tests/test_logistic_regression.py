import numpy as np

from main import cost, gradient_descent, sigmoid


def test_sigmoid_outputs_probabilities_in_range():
    probabilities = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.isclose(probabilities[1], 0.5)


def test_gradient_descent_reduces_logistic_cost():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    initial_cost = cost(len(y), y, sigmoid(np.zeros(len(y))))

    weights, bias, history = gradient_descent(
        X, y, alpha=0.1, num_iters=1000
    )
    final_probabilities = sigmoid(X @ weights + bias)
    final_cost = cost(len(y), y, final_probabilities)

    assert final_cost < initial_cost
    assert history.shape == (1000,)
