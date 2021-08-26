import os
import streamlit as st
import sqlite3
import hashlib
import time
import numpy as np

# Security
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management
USER_DB = ".userdata.db"
conn = sqlite3.connect(USER_DB)
c = conn.cursor()


def cache_state(state):
    return state


# DB  Functions
def create_usertable():
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")


def add_userdata(username, password):
    c.execute(
        "INSERT INTO userstable(username,password) VALUES (?,?)", (username, password)
    )
    conn.commit()


@st.cache
def login_user(username, password):
    c.execute(
        "SELECT * FROM userstable WHERE username =? AND password = ?",
        (username, password),
    )
    data = c.fetchall()
    return data


def view_all_users():
    c.execute("SELECT * FROM userstable")
    data = c.fetchall()
    return data


########################################################################


def main():
    st.title("Analysis App for the stress study in 2021")
    st.write("---")
    text = st.empty()
    text.warning("Please login before accessing the data")
    user_box = st.sidebar.empty()
    pwd_box = st.sidebar.empty()
    submit = st.sidebar.empty()
    user_box.text_input("Username", "", key="username")
    pwd_box.text_input("Password", "", key="password", type="password")
    st.session_state.checkbox_state = submit.checkbox("Login")
    if st.session_state.checkbox_state:
        # if password == '12345':
        create_usertable()
        hashed_pswd = make_hashes(st.session_state.password)
        result = login_user(
            st.session_state.username,
            check_hashes(st.session_state.password, hashed_pswd),
        )
        if result:
            user_box.empty()
            pwd_box.empty()
            submit.empty()
            text.success("Log in as {}".format(st.session_state.username))
            text.empty()
            logout_holder = st.sidebar.empty()
            logout = logout_holder.button("Log out")
            if logout:
                logout_holder.empty()
                text.success("Log out. Please refresh the page.")
                return

            choice = st.sidebar.selectbox("Data", ["a", "b", "c"])
            if choice == "a":
                st.dataframe(np.random.randint(0, 255, 300 * 300).reshape(300, 300))

            # start your code here

        else:
            text.warning("Invalid Username or Password")


if __name__ == "__main__":
    main()
