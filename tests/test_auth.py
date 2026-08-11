from app.auth import login, signup, verify_token


def test_signup_and_login_success():
    assert signup("alice", "s3cret") is True
    token = login("alice", "s3cret")
    assert token is not None
    assert isinstance(token, str)


def test_login_wrong_password_fails():
    signup("bob", "correct-password")
    token = login("bob", "wrong-password")
    assert token is None


def test_verify_token_roundtrip():
    signup("carol", "hunter2")
    token = login("carol", "hunter2")
    username = verify_token(token)
    assert username == "carol"
    assert verify_token("not-a-real-token") is None
