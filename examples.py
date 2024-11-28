import numpy as np

tasks = {
        'random_fct': {
        'dimension': 50,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: random.choice([-1, 1])
    },
    'fullparity': {
        'dimension': 50,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :].prod(axis=1)
    },
    'fullparity_d150': {
        'dimension': 150,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :].prod(axis=1)
    },
    'fullparity_d100': {
        'dimension': 100,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :].prod(axis=1)
    },
    'fullparity_d200': {
        'dimension': 200,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :].prod(axis=1)
    },
    'fullparity_d250': {
        'dimension': 250,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :].prod(axis=1)

    'parity3': {
        'dimension': 50,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: X[:, :3].prod(axis=1)
    } 
        , 
    'leapCSQ3': {
        'dimension': 50,
        'valid_size': 8192,
        'test_size': 8192,
        'target_function': lambda X: 1/4*(0.5*X[:, :3].prod(axis=1) + 1.5*X[:, [0, 1, 3]].prod(axis=1) + X[:, [0, 2, 3]].prod(axis=1) + X[:, 1:4].prod(axis=1))
    }


}
