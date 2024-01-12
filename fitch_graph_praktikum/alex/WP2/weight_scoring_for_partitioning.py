def average_weight_scoring(left: [], right: [], relations: dict[tuple[int, int], float]):
    score = 0
    counter = 0
    for l in left:
        for r in right:
            if (l, r) in relations:
                score += relations[(l, r)]
                counter += 1
    score = score / counter if counter > 0 else 0
    return score


def sum_weight_scoring(left: [], right: [], relations: dict[tuple[int, int], float]):
    score = 0
    for l in left:
        for r in right:
            if (l, r) in relations:
                score += relations[(l, r)]
    return score
