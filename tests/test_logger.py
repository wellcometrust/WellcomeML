from wellcomeml.logger import logger


def test_logging():
    """Tests the logger name"""
    assert logger.name == 'wellcomeml.logger'
