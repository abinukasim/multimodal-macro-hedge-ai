import re
_whitespace = re.compile(r"\s+")
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _whitespace.sub(" ", s)       # collapse whitespace/newlines
    if len(s) > 4000:                 # trim very long entries
        s = s[:4000]
    return s
