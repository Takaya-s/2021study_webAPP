import os
import sqlite3
import hashlib


# DB Management
USER_DB = ".userdata.db"
conn = sqlite3.connect(USER_DB)
c = conn.cursor()


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# only for create database
def initiate_a_user(username, password=None):
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")
    if password is None:
        password = os.urandom(4).hex()

    c.execute(
        "INSERT INTO userstable(username,password) VALUES (?,?)",
        (username, make_hashes(password)),
    )
    conn.commit()
    with open("./secret.txt", "w") as file:
        file.write(f"username: {username}\npassword: {password}\n")
        print("write at secret.txt")


if __name__ == "__main__":
    username = "admin"
    password = "123"
    initiate_a_user(username, password)
