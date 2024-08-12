import h3


def token2centroid_h3_yx(lat, long):
    # returns centroid of the hexagon
    token = h3.geo_to_h3(lat, long, 10)
    return token


def tokenizeTrajectories(data, mode):
    result_lines = []

    for item in data:
        trajectory_id = item["id"]
        trajectory_points = item["trajectory"].split(",")
        summary_tokens, trajectory_tokens = [], []
        # For each line in data (trajectory, summary) apply the defined tokenization function
        trajectory_tokens = [
            token2centroid_h3_yx(float(lat), float(long))
            for lat, long in (point.split() for point in trajectory_points)
        ]
        if mode != "generation_training":
            summary_points = item["summary"].split(",")

            summary_tokens = [
                token2centroid_h3_yx(float(lat), float(long))
                for lat, long in (point.split() for point in summary_points)
            ]
        if mode == "summarization_training":
            result_line = f'<original> {" ".join(trajectory_tokens)} <end> <summary> {" ".join(summary_tokens)}<end>'
        elif mode == "summarization_testing":
            result_line = f'<original> {" ".join(trajectory_tokens)} <end> <summary> {" ".join(summary_tokens)}'
        elif mode == "generation_training":
            result_line = f'{" ".join(trajectory_tokens)}'
        result_lines.append(result_line)

    return result_lines


def writeTokenizedTrajectories(filepath: str, data):
    with open(filepath, "w") as file:
        for line in data:
            file.write(line + "\n")
