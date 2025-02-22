def split_list(lst, max_size: int = 41666):
    num_parts = (len(lst) + max_size - 1) // max_size
    avg_size = len(lst) // num_parts

    parts = []
    for i in range(num_parts):
        start = i * avg_size
        end = start + avg_size if i != num_parts - 1 else len(lst)
        parts.append(lst[start:end])

    return parts
