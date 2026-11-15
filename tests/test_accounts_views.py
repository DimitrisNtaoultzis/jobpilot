import pytest
from django.urls import reverse
from pytest_django.asserts import assertContains, assertRedirects

STRONG_PASSWORD = "a-Strong-pass-123"


@pytest.mark.django_db
def test_signup_creates_the_user_and_logs_them_in(client, django_user_model):
    response = client.post(
        reverse("accounts:signup"),
        {"username": "carol", "password1": STRONG_PASSWORD, "password2": STRONG_PASSWORD},
    )

    assertRedirects(response, reverse("home"))
    assert django_user_model.objects.get(username="carol").check_password(STRONG_PASSWORD)
    assertContains(client.get(reverse("home")), "Hello, carol")


@pytest.mark.django_db
def test_signup_rejects_a_weak_password(client, django_user_model):
    response = client.post(
        reverse("accounts:signup"),
        {"username": "carol", "password1": "1234", "password2": "1234"},
    )

    assert response.status_code == 200  # form shown again with errors
    assert "password2" in response.context["form"].errors
    assert not django_user_model.objects.filter(username="carol").exists()


@pytest.mark.django_db
def test_login_with_correct_credentials_redirects_to_next(client, alice):
    cv_list = reverse("jobs:cv_list")

    response = client.post(
        f"{reverse('accounts:login')}?next={cv_list}", {"username": "alice", "password": "pw"}
    )

    assertRedirects(response, cv_list)


@pytest.mark.django_db
def test_login_with_wrong_password_fails(client, alice):
    response = client.post(reverse("accounts:login"), {"username": "alice", "password": "wrong"})

    assert response.status_code == 200
    assert response.context["form"].errors
    assert "_auth_user_id" not in client.session  # where Django stores the logged-in user


def test_logout_rejects_get(client):
    assert client.get(reverse("accounts:logout")).status_code == 405


@pytest.mark.django_db
def test_logout_with_post_logs_out(client, alice):
    client.force_login(alice)

    response = client.post(reverse("accounts:logout"))

    assertRedirects(response, reverse("home"))
    assert "_auth_user_id" not in client.session
