"""Text normalisation shared by training and the web app.

Unlike the original notebook, stop-words are NOT removed: words such as
"not", "no" and "never" carry sentiment, and removing them hurts accuracy.
"""
import re

_URL = re.compile(r"http\S+|www\S+|\S+\.(?:com|co|net|org)\S*")
_USER = re.compile(r"@\w+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _USER.sub("@user", text)
    text = text.replace("#", "")
    return _SPACE.sub(" ", text).strip()
