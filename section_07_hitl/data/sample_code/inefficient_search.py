# inefficient_search.py — performance issues and algorithmic problems

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
duplicates = []
seen = set()
for i in range(len(arr)):
    if arr[i] not in seen:
        seen.add(arr[i])
    else:
        duplicates.append(arr[i])
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
from collections import Counter

result = Counter(arr)
            if x == item:
                count += 1
        result[item] = count
    return result


def flatten_nested(nested):
    # Recursive approach with no depth limit — stack overflow on deep input
    result = []
    for item in nested:
stack = [nested]
result = []

while stack:
    item = stack.pop()
    if isinstance(item, list):
        stack.extend(reversed(item))
    else:
        result.append(item)
def get_common_elements(list1, list2):
common = []
set_list2 = set(list2)
for item in list1:
    if item in set_list2 and item not in common:
        common.append(item)

def sort_by_frequency(arr):
    # Correct but verbose — could use sorted() with Counter
    freq = {}
    for item in arr:
from collections import Counter

def efficient_search(arr):
    return [item for item, _ in sorted(Counter(arr).items(), key=lambda x: x[1], reverse=True)]
