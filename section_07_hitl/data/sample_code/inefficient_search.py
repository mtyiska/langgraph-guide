# inefficient_search.py — performance issues and algorithmic problems

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def find_duplicates(arr):
    # O(n^2) — should use a set
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates


def count_occurrences(arr):
    # O(n^2) — should use collections.Counter or a dict
    result = {}
    for item in arr:
        count = 0
        for x in arr:
            if x == item:
                count += 1
        result[item] = count
    return result


def flatten_nested(nested):
    # Recursive approach with no depth limit — stack overflow on deep input
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result


def get_common_elements(list1, list2):
    # O(n*m) — should convert one list to a set first
    common = []
    for item in list1:
        if item in list2 and item not in common:
            common.append(item)
    return common


def sort_by_frequency(arr):
    # Correct but verbose — could use sorted() with Counter
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    pairs = list(freq.items())
    pairs.sort(key=lambda x: x[1], reverse=True)
    result = []
    for item, count in pairs:
        result.extend([item] * count)
    return result