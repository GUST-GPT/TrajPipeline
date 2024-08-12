import h3
import warnings
import pickle
import math
import numpy as np
from tqdm import tqdm
from random import random
import os
import json

warnings.filterwarnings("ignore")


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_bearing(pointA, pointB):
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    )

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


class BERTImputer(object):
    h3_clusters = None
    h3_kmeans = None

    def __init__(self):
        # adjust data dir as needed.
        # data_dir = "."
        data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{data_dir}/h3_clusters.pkl", "rb") as file:
            self.h3_clusters = pickle.load(file)
        with open(
            f"{data_dir}/h3_kmeans_clustering_all_models_precise.pkl", "rb"
        ) as file:
            self.h3_kmeans = pickle.load(file)

    def token2point_h3_centroid(self, token, previous_point=None):
        y, x = h3.h3_to_geo(token)
        return Point(x, y)

    def token2point_data_centroid(self, token, previous_point=None):
        if token in self.h3_clusters:
            cluster = self.h3_clusters[token]
            x, y = cluster["x"], cluster["y"]
            return Point(x, y)
        else:
            return self.token2point_h3_centroid(token, None)

    def token2point_cluster_centroid(self, token, previous_point):
        c = self.token2point_data_centroid(token, None)

        if token not in self.h3_kmeans:
            return c

        if not previous_point:
            return c

        if token in self.h3_clusters and self.h3_clusters[token]["current_count"] <= 20:
            return c

        angle = calculate_bearing((previous_point.y, previous_point.x), (c.y, c.x))
        m, means = self.h3_kmeans[token]
        x, y, _ = means[m.predict(np.array([angle]).reshape(-1, 1))][0]
        return Point(x, y)


def readTrajectoriesFile(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines


def detokenizeLine(line, bertImputerInstance, mode):
    elements = line.split()
    detokenized_trajectory = []
    previous_point = None
    is_summary = False
    if mode == "summarization_testing":
        detokenized_trajectory.append("<original>")
    for i, element in enumerate(elements):
        if element == "<end>":
            if mode == "summarization_testing":
                detokenized_trajectory.append(" <end>")
            if is_summary:
                break
            if mode == "summarization_testing":
                detokenized_trajectory.append(" <summary>")
            is_summary = True
        elif element not in ("<original>", "<summary>", "<end>", "<pad>"):
            if h3.h3_is_valid(element):
                point = bertImputerInstance.token2point_cluster_centroid(
                    element, previous_point
                )
                previous_point = point
                detokenized_trajectory.append(f"{round(point.y,6)} {round(point.x,6)}")
                # Add a comma if it's not the last element
                if i < len(elements) - 1:
                    detokenized_trajectory.append(",")
    result = "".join(detokenized_trajectory)

    return result


def detokenizeTrajectories(input_file, bertImputerInstance, mode):
    lines = readTrajectoriesFile(input_file)
    detokenizedTrajectories = []
    for line in lines:
        detokenized_line = detokenizeLine(line, bertImputerInstance, mode)
        # print(detokenized_line)
        detokenizedTrajectories.append(detokenized_line)
    return detokenizedTrajectories


def writeDetokenizedTrajectories(detokenized_trajectories, output_file, mode):
    detokenized_data = []
    for idx, detokenized in enumerate(detokenized_trajectories, start=1):
        detokenized = detokenized.strip()
        # print(detokenized)
        trajectory_dict = {}
        if mode == "summarization_testing":
            if detokenized.startswith("<original>"):
                parts = detokenized.split(", <end> <summary>")

                original_points = parts[0].replace("<original>", "").strip()
                # print(original_points)
                trajectory_dict = {
                    "id": str(idx),
                    "trajectory": original_points,
                }
                if len(parts) > 1:
                    summary_points = parts[1].replace(", <end>", "").strip()
                    trajectory_dict = {
                        "id": str(idx),
                        "trajectory": original_points,
                        "summary": summary_points,
                    }
                    # trajectory_dict["summary"] = summary_points
        else:
            trajectory_dict = {
                "id": str(idx),
                "trajectory": detokenized,
            }
        detokenized_data.append(trajectory_dict)

    with open(output_file, "w") as f:
        json.dump(detokenized_data, f, indent=4)
