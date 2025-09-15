from src.utils.http import create_retry_session


def test_create_retry_session():
    s = create_retry_session()
    # Ensure adapters are mounted
    assert "http://" in s.adapters
    assert "https://" in s.adapters
