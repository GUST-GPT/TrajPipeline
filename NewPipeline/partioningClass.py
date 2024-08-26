"""Partioning Module Definition"""

import os
import json
from utilFunctions import load_metadata, load_tokenized_trajectories


class PartitioningModule:
    """
    A class to manage a hierarchical pyramid structure for storing and updating models
    based on trajectory datasets. The pyramid structure is defined by levels and cells,
    with each cell potentially containing a model. The class handles the initialization
    of the pyramid structure, loading configuration parameters, and updating the model
    repository with new datasets.

    Attributes:
        config_file (str): The path to the JSON configuration file.
        H (int): The number of levels in the pyramid.
        L (int): The number of cells per level.
        pyramid (dict): The hierarchical pyramid structure where each level contains cells.
        model_repo_dir (str): The directory where model files are stored, structured as
        height_level_index.
    """

    def __init__(self, models_repo_path):
        """
        Initializes the PartitioningModule with configurations read from a JSON file.
        """
        config_file = os.path.join(models_repo_path, "pyramidConfig.json")
        self.pyramid_path = os.path.join(models_repo_path, "partioningPyramid.json")
        self.config_file = config_file
        self.pyramid_height = 5
        self.pyramid_levels = 3
        self.build_pyramid_flag = False
        self.pyramid = {}
        self.model_repo_dir = models_repo_path
        self.tokens_threshold_per_cell = 20000
        self.load_config()
        if self.build_pyramid_flag:
            self.build_pyramid()
        else:
            self.load_pyramid()
        print(self.pyramid)

    def load_config(self):
        """
        Loads the configuration parameters H and L from the JSON file.
        """
        with open(self.config_file, "r") as file:
            config = json.load(file)
            self.pyramid_height = config.get("H", 5)  # Default to 5 if not specified
            self.pyramid_levels = config.get("L", 3)  # Default to 3 if not specified
            self.build_pyramid_flag = config.get("build_pyramid_from_scratch")

    def build_pyramid(self):
        """
        Builds the pyramid data structure for the models repository.
        """
        self.pyramid = {}  # Reset the pyramid structure
        for h in range(self.pyramid_height + 1):
            self.pyramid[h] = self._generate_cells(h)
        # Save the pyramid to the JSON file
        with open(self.pyramid_path, "w") as file:
            json.dump(self.pyramid, file, indent=4)

    def load_pyramid(self):
        """
        Loads the pyramid data structure from the JSON file.
        """
        if os.path.exists(self.pyramid_path):
            with open(self.pyramid_path, "r") as file:
                self.pyramid = json.load(file)
        else:
            raise FileNotFoundError(f"Pyramid file not found at {self.pyramid_path}")

    def _generate_cells(self, h):
        """
        Generates cells for a given height h.
        """
        num_cells = 4**h
        cells = {}
        for i in range(num_cells):
            cells[i] = {
                "height": h,
                "index": i,
                "bounds": self._calculate_bounds(h, i),
                "occupied": False,
                "model_path": None,
                "num_tokens": 0,
            }
        return cells

    def _calculate_bounds(self, h, index):
        """
        Calculates the bounds for a cell at a given height and index.
        """
        total_cells = 4**h
        cell_size = 1 / (4**h)
        lat_start = (index // int(total_cells**0.5)) * cell_size
        lon_start = (index % int(total_cells**0.5)) * cell_size
        return (lat_start, lat_start + cell_size, lon_start, lon_start + cell_size)

    def update_repository(self, data_path, metadata_path):
        """
        Updates the model repository with a model.

        Args:
            data_path: The path for the new trajectory dataset to update the repository.
            metadata_path: The path for metadata of the new trajectory dataset to update the repository.
        """
        new_trajectory_dataset = load_tokenized_trajectories(data_path)
        new_trajectory_dataset_metadata = load_metadata(metadata_path)
        num_tokens = new_trajectory_dataset_metadata.get("total_number_of_tokens")
        # Calculate the minimum bounding rectangle of all trajectories
        min_bounding_rectangle = self._calculate_mbr(new_trajectory_dataset)

        # Find the smallest cell that fully encloses this minimum bounding rectangle
        target_cell = self._find_enclosing_cell(min_bounding_rectangle)

        # Update the model repository
        if target_cell:
            # Only add new model to cell, if #tokens is at least k*4**(H-l)
            # @YoussefDo: Make sure this is correct, to get the dataset as number of tokens

            h = target_cell["height"]
            if num_tokens >= (
                self.tokens_threshold_per_cell * 4 ** (self.pyramid_height - h)
            ):
                self._update_cell_with_model(
                    target_cell, new_trajectory_dataset, num_tokens
                )
            else:
                raise ValueError("Not sufficient data to train a model.")
        else:
            raise ValueError(
                "No suitable cell found for the given trajectories in the pyramid."
            )

    def _calculate_mbr(self, trajectories):
        """
        Calculates the minimum bounding rectangle (MBR) for a set of trajectories.

        Args:
            trajectories (list of list of tuples): List of trajectories, where each trajectory is a list of (lat, lon) tuples.

        Returns:
            tuple: A tuple representing the MBR (min_lat, max_lat, min_lon, max_lon).
        """
        min_lat = min_lon = float("inf")
        max_lat = max_lon = float("-inf")

        for trajectory in trajectories:
            for point in trajectory:
                lat, lon = point
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)

        return (min_lat, max_lat, min_lon, max_lon)

    def _find_enclosing_cell(self, bounding_rectangle):
        """
        Finds the smallest cell that fully encloses the given bounding rectangle.
        """
        for h in reversed(range(self.pyramid_height + 1)):
            for i, cell in self.pyramid[h].items():
                if self._is_bounding_rectangle_enclosed(
                    bounding_rectangle, cell["bounds"]
                ):
                    return cell
        return None

    def _is_bounding_rectangle_enclosed(self, rectangle, cell_bounds):
        """
        Checks if a bounding rectangle is fully enclosed within the cell bounds.
        """
        lat_min, lat_max, lon_min, lon_max = rectangle
        cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max = cell_bounds
        return (
            lat_min >= cell_lat_min
            and lat_max <= cell_lat_max
            and lon_min >= cell_lon_min
            and lon_max <= cell_lon_max
        )

    def _update_cell_with_model(self, cell, dataset, num_tokens):
        """
        Updates the cell with a new model and stores it in the models repository.
        """
        h = cell["height"]
        index = cell["index"]
        cell_path = os.path.join(self.model_repo_dir, f"{h}_{index}")

        # Create the directory if it doesn't exist
        if not os.path.exists(cell_path):
            os.makedirs(cell_path)

        # Define the model path
        cell["model_path"] = cell_path
        cell["occupied"] = True
        # @YoussefDo: I need to think about the logic of integrating two datasets together
        # and linking the dataset in the trajectory story to this cell
        cell["num_tokens"] = num_tokens

        # @YoussefDo: Implement logic to train and save the model in the cell_path
        # For example:
        # with open(os.path.join(cell_path, 'model.pkl'), 'wb') as f:
        #     pickle.dump(model, f)
        pass

    def find_proper_model(self, test_data):
        """
        Used in the testing mode to find the proper model for the passed query data

        Args:
            test_data: trajectory test data used to find proper model and load it in memory
        """
        # Calculate the minimum bounding rectangle of all trajectories
        min_bounding_rectangle = self._calculate_mbr(test_data)

        # Find the smallest cell that fully encloses this minimum bounding rectangle
        target_cell = self._find_enclosing_cell(min_bounding_rectangle)
        if target_cell:  # Then we found a cell that encloses this trajectory data
            # @YoussefDo: need to load the model here
            model_path = target_cell["model_path"]
            # Tensorflow load model
            print(model_path)
        else:
            raise ValueError(
                "No proper model found for requested trajectory query data"
            )
        pass
