import os


def lowest_number_for_dir(dir):
    if not os.path.exists(dir):
        return 0
    basenames = [f.split(".")[0] for f in os.listdir(dir)]
    numbers = [int(b) for b in basenames if b.isnumeric()]
    return max(numbers) + 1 if numbers else 0


def get_ground_truth(comment: str):
    gt = comment.strip().split()
    assertion_error_message = f"Incorrect value specification in sgf {' '.join(gt)}"
    assert len(gt) <= 2, assertion_error_message
    if len(gt) == 0:
        return None
    if gt[0].isnumeric():
        value = float(gt[0])
    else:
        value = 0
    if len(gt) > 1 or not gt[0].isnumeric():
        quotient = gt[-1].split('/')
        assert len(quotient) == 2, assertion_error_message
        value += float(quotient[0]) / float(quotient[1])
    return value


def almost_equal(val1, val2, threshold=0.00001):
    return abs(float(val1 - val2)) <= threshold
