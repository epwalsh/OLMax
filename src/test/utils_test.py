import olmax.utils as utils


def test_running_average():
    avg = utils.RunningAverage(0.0)
    assert avg.update(2.0) == 2.0
    assert avg.update(3.0) == 2.5
    assert avg.update(2.5) == 2.5
