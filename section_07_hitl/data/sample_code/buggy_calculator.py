# buggy_calculator.py — intentionally contains bugs for the review agent to find

def divide(a, b):
    # Bug: no zero division check
    return a / b


def average(numbers):
    # Bug: crashes on empty list
    return sum(numbers) / len(numbers)


def factorial(n):
    # Bug: no base case guard for negative numbers
    if n == 0:
        return 1
    return n * factorial(n - 1)


def find_max(items):
    # Bug: returns None silently on empty list instead of raising
    if not items:
        return None
    max_val = items[0]
    for item in items:
        if item > max_val:
            max_val = item
    return max_val


def celsius_to_fahrenheit(c):
    # Bug: wrong formula — should be (c * 9/5) + 32
    return c * 9 + 32


def is_palindrome(s):
    # Style: overly verbose, also doesn't handle case or spaces
    reversed_s = ""
    for char in s:
        reversed_s = char + reversed_s
    if s == reversed_s:
        return True
    else:
        return False


def count_vowels(text):
    # Performance: rebuilds string on every character
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count = count + 1
    return count