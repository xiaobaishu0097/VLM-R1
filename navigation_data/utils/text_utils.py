import random
import re
import string


def split_string_at_capital(input_str: str) -> str:
    return re.sub(r"(?<=[a-z])([A-Z])", r" \1", input_str)


def replace_last_comma(s: str) -> str:
    parts = s.rsplit(", ", 1)
    return " and ".join(parts) if len(parts) > 1 else s


def remove_last_comma(s: str) -> str:
    parts = s.rsplit(", ", 1)
    return "".join(parts) if len(parts) > 1 else s


def get_indefinite_article(word: str) -> str:
    vowels = ("a", "e", "i", "o", "u", "A", "E", "I", "O", "U")
    if word.startswith(vowels):
        return f"an {word}"
    else:
        return f"a {word}"


def concat_list(items: list) -> str:
    # Check if the list is empty or contains only one element
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]

    # Join all elements except the last one with a comma, then add 'and' before the last element
    return ", ".join(items[:-1]) + ", and " + items[-1]


def generate_random_string():
    # Randomly choose the length between 3 and 63
    length = random.randint(3, 63)
    # Generate a random string of the chosen length
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=length)
    )
    return random_string
