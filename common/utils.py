import os


def lowest_number_for_dir(dir):
    if not os.path.exists(dir):
        return 0
    basenames = [f.split(".")[0] for f in os.listdir(dir)]
    numbers = [int(b) for b in basenames if b.isnumeric()]
    return max(numbers) + 1 if numbers else 0