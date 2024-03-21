def calculate_difference(a: list[int]) -> list[int]:
    difference = []
    for i in range(len(a)):
        for j in range(i,len(a)):
            if i == j:
                continue
            difference.append(a[j] - a[i])
    print(f"size of the list: {len(difference)}")
    print(sorted(difference))
    return difference



if __name__ == "__main__":
    calculate_difference([0, 3, 5, 9, 16, 17])
    calculate_difference([0, 1, 4, 9, 15, 22, 32, 34])
    calculate_difference([0, 1, 6, 10, 23, 26, 34, 41, 53, 55])