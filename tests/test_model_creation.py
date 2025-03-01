import biobuddy


def test_version():
    assert biobuddy.__version__ == "0.1.0"


def test_force_test():
    assert 2 + 2 == 4
