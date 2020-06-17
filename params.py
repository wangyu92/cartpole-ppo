from types import SimpleNamespace

data = {
    'NUM_AGENTS': 2,
    'HORIZON': 4096,
    'NUM_MINIBATCH': 2,
    'NUM_EPOCH': 8,
    'CLIPRANGE': 0.2,
    'GAMMA': 0.99,
    'LAMBDA': 0.95,

    'ENT_MAX': 0.01,
    'ENT_MIN': 0.01,
    'ENT_STEP': 5e+4,
    'LR': 1e-4,
    'PLOSS_TYPE': 'truly',
    'VLOSS_TYPE': 'clip',

    # Truly PPO
    'RB_ALPHA': 0.3,
    'TR_DELTA': 0.035,
    'TRULY_ALPHA': 0.03,
    'TRULY_DELTA': 5.,

    # test
    'TEST_ITER': 20,

    # Train status
    'NUM_UPDATE': 0,
    'NUM_EPISODE': 0,
}

def get():
    params = SimpleNamespace(**data)
    return params