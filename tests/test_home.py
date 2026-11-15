import pytest
from django.urls import reverse
from pytest_django.asserts import assertContains, assertNotContains


def test_home_anonymous_sees_login_link(client):
    response = client.get(reverse("home"))

    assertContains(response, "Log in")
    assertNotContains(response, "Log out")


@pytest.mark.django_db
def test_home_logged_in_sees_greeting(client, django_user_model):
    user = django_user_model.objects.create_user(username="alice", password="pw")
    client.force_login(user)

    response = client.get(reverse("home"))

    assertContains(response, "Hello, alice")
    assertContains(response, "Log out")
