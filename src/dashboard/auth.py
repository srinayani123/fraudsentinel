"""Supabase authentication for FraudSentinel.

Handles email/password sign-in, sign-up, Google OAuth (PKCE flow), and
password reset through Supabase.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def _get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None, None
    return url, key


@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    url, key = _get_supabase_config()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None


def sign_in_with_password(email: str, password: str) -> tuple[bool, str]:
    client = get_supabase_client()
    if not client:
        return False, "Supabase not configured"
    try:
        response = client.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = response.user.email
            st.session_state["user_id"] = response.user.id
            st.session_state["user_role"] = "Fraud Analyst"
            st.session_state["access_token"] = response.session.access_token
            st.session_state["_just_signed_in"] = True
            return True, "Signed in successfully"
        return False, "Invalid credentials"
    except Exception as e:
        msg = str(e)
        if "Invalid login credentials" in msg:
            return False, "Invalid email or password"
        if "Email not confirmed" in msg:
            return False, "Please verify your email before signing in"
        return False, f"Sign in failed: {msg}"


def sign_up_with_password(email: str, password: str, full_name: str = "") -> tuple[bool, str]:
    client = get_supabase_client()
    if not client:
        return False, "Supabase not configured"
    try:
        response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}} if full_name else {},
        })
        if response.user:
            if response.session:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = response.user.email
                st.session_state["user_id"] = response.user.id
                st.session_state["user_role"] = "Fraud Analyst"
                st.session_state["access_token"] = response.session.access_token
                st.session_state["_just_signed_in"] = True
                return True, "Account created and signed in"
            return True, "Account created — check your email to verify"
        return False, "Sign up failed"
    except Exception as e:
        msg = str(e)
        if "already registered" in msg.lower():
            return False, "An account with this email already exists"
        if "weak" in msg.lower() or "password" in msg.lower():
            return False, "Password is too weak"
        return False, f"Sign up failed: {msg}"


def sign_in_with_google() -> Optional[str]:
    client = get_supabase_client()
    if not client:
        return None
    try:
        response = client.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {"redirect_to": "http://localhost:8501"},
        })
        return response.url
    except Exception as e:
        st.error(f"Google sign-in setup failed: {e}")
        return None


def send_password_reset(email: str) -> tuple[bool, str]:
    client = get_supabase_client()
    if not client:
        return False, "Supabase not configured"
    try:
        client.auth.reset_password_email(email, {"redirect_to": "http://localhost:8501"})
        return True, "Password reset email sent. Check your inbox."
    except Exception as e:
        return False, f"Reset failed: {e}"


def sign_out():
    client = get_supabase_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    for k in ("logged_in", "user_email", "user_id", "user_role", "access_token", "auth_view", "_just_signed_in"):
        st.session_state.pop(k, None)


def handle_oauth_callback():
    """Handle OAuth PKCE callback — exchange ?code=... for a session.

    When Google redirects back from Supabase OAuth, the URL contains
    ?code=<auth_code>. We exchange this code for a session via
    `exchange_code_for_session`, then set Streamlit session state and rerun.

    If the user is already logged in (page reload after auth), we silently
    clear the leftover query param.
    """
    # Already signed in — just clear leftover query params silently
    if st.session_state.get("logged_in"):
        if "code" in st.query_params:
            st.query_params.clear()
        return

    if "code" not in st.query_params:
        return

    auth_code = st.query_params.get("code")
    if not auth_code:
        return

    client = get_supabase_client()
    if not client:
        return

    try:
        response = client.auth.exchange_code_for_session({"auth_code": auth_code})

        if response and response.user and response.session:
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = response.user.email
            st.session_state["user_id"] = response.user.id
            st.session_state["user_role"] = "Fraud Analyst"
            st.session_state["access_token"] = response.session.access_token
            st.session_state["_just_signed_in"] = True

            st.query_params.clear()
            st.rerun()
        else:
            st.query_params.clear()
    except Exception:
        # Code already used or expired — silently clear so user sees clean login
        st.query_params.clear()


def is_supabase_configured() -> bool:
    url, key = _get_supabase_config()
    return bool(url and key)
