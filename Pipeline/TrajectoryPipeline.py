import json
from typing import List, Dict
from TrajPipeline.Pipeline.Tokenization.tokenization import *
from TrajPipeline.Pipeline.Detokenization.detokenization import *
import os
import subprocess
import logging


class TrajectoryPipeline:
    def __init__(self):
        self.trajectories = []
        self.summaries = []
        self.params = {}
        self.mode, self.city, self.input_file_path = "", "", ""
        self.trajectories_length, self.trajectories_count = 0, 0
        self.data = []
        # Get the directory of the pipeline
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bert_imputer_instance = BERTImputer()
        self.tokenized_trajectories, self.detokenized_trajectories = [], []

    def load_data(self):
        with open(self.input_file_path, "r") as file:
            self.data = json.load(file)
            if isinstance(self.data, list):
                if "summary" in self.data[0]:
                    self.trajectories = [item["trajectory"] for item in self.data]
                    self.summaries = [item["summary"] for item in self.data]
                else:
                    self.trajectories = [item["trajectory"] for item in self.data]
            else:
                raise ValueError("Invalid data format")

    def load_params(self, filepath: str):
        with open(filepath, "r") as file:
            params = json.load(file)

        self.mode = params.get("mode", self.mode)
        self.city = params.get("city", self.city)
        self.input_file_path = params.get("input_path", self.input_file_path)
        # The only case we don't need to load a dataset is when a user wants generation testing
        if self.mode != "generation_testing":
            self.load_data()
        self.trajectories_length = params.get(
            "trajectories_length", self.trajectories_length
        )
        self.trajectories_count = params.get(
            "trajectories_count", self.trajectories_count
        )
        print("Params loaded successfully...")

    def save_data(self, filepath: str, data: List[Dict[str, str]]):
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    def save_params(self, filepath: str, params: Dict[str, str]):
        with open(filepath, "w") as file:
            json.dump(params, file, indent=4)

    def tokenizationModule(self):
        # Go to Tokenization directory do the tokenization based on the scripts over there to the data loaded here.
        tokenized_trajectories_path = os.path.join(
            self.script_dir, "Tokenization/tokenizedTrajectories.txt"
        )
        self.tokenized_trajectories = tokenizeTrajectories(
            data=self.data, mode=self.mode
        )
        writeTokenizedTrajectories(
            filepath=tokenized_trajectories_path, data=self.tokenized_trajectories
        )
        print(f"Tokenization complete to {tokenized_trajectories_path}")
        # Now I wrote the tokenized data, and I also have it stored in my variable self.tokenized_trajectories.

    def deTokenizationModule(self):
        # Go to Detokenization directory do the detokenization based on the scripts over there to the data loaded here.

        tokenized_trajectories_path = os.path.join(
            self.script_dir, "Detokenization/tokenizedTrajectories.txt"
        )
        deTokenized_trajectories_path = os.path.join(
            self.script_dir, "Detokenization/detokenizedTrajectories.json"
        )
        self.detokenized_trajectories = detokenizeTrajectories(
            tokenized_trajectories_path,
            self.bert_imputer_instance,
            mode=self.mode,
        )
        # print("Hello")
        # print(self.detokenized_trajectories)
        writeDetokenizedTrajectories(
            detokenized_trajectories=self.detokenized_trajectories,
            output_file=deTokenized_trajectories_path,
            mode=self.mode,
        )
        print("Detokenization complete. Data saved to", deTokenized_trajectories_path)

    def fineTuningModule(self):
        # Go to finetuning directory and see training params over there, the user can edit them to tune their model.
        # we should read the params from there and tune the model accordingly.
        pass

    def spatialConstraintsModule(self):
        # Go to spatialConstrains directory and see the constraints over there, the user can edit them to adjust their model.
        # we should read the constraints from there and adjust the model accordingly to apply these rules.
        pass

    def modelsRepository(self):
        transformers_path = "/speakingTrajectories/Transformers"
        # This where we will dump the output, i.e. from generated_trajecories and simplified_trajectories from Transformers
        # These will still be tokenized, as output of Transformers is tokens, so we pass this to Detokenization and we are done.
        # This is only needed in the "Testing" Phase
        final_trajectories_path = os.path.join(
            self.script_dir, "Detokenization/tokenizedTrajectories.txt"
        )
        working_directory = "/speakingTrajectories/Transformers/nanoGPT"
        training_model_path = "train.py"

        # Go to models repo directory and save/load the model there. We need to apply the pyramid idea, but not now.
        if self.mode == "summarization_training":
            # I need to pass the given self.tokenized_trajectories to /speakingTrajectories/Transformers/nanoGPT/data/trajectorySummary/input.txt
            # Then run prepare.py and start the training process with the new finetuning arch. and (spatial constrainsts?).
            configurations_model_path = "config/train_trajectory_summary.py"
            input_trajectories_path = os.path.join(
                transformers_path, "nanoGPT/data/newTrajectorySummary"
            )

            with open(os.path.join(input_trajectories_path, "input.txt"), "w") as file:
                for line in self.tokenized_trajectories:
                    file.write(line + "\n")

            try:
                result = subprocess.run(
                    ["python", os.path.join(input_trajectories_path, "prepare.py")],
                    capture_output=True,
                    text=True,
                )
                print("Script output:", result.stdout)
                print("Data prepared successfully for the model...")
            except subprocess.CalledProcessError as e:
                print("Error preparing data for the model:", e)
            print("Starting Model Training Now...")
            # Assuming I have the new model architecure
            os.chdir(working_directory)
            try:
                process = subprocess.run(
                    ["python", training_model_path, configurations_model_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print("Script output:", process.stdout)
                print("Trained model for summarization successfully.")
            except subprocess.CalledProcessError as e:
                print("Error training model for summarization:", e)
                print("Script error output:", e.stderr)
        elif self.mode == "generation_training":
            # I need to pass the given self.tokenized_trajectories to /speakingTrajectories/Transformers/nanoGPT/data/trajectorySummary/input.txt
            # Then run prepare.py and start the training process with the new finetuning arch. and (spatial constrainsts?).
            configurations_model_path = "config/train_trajectory.py"
            input_trajectories_path = os.path.join(
                transformers_path, "nanoGPT/data/newTrajectoryGeneration"
            )

            with open(os.path.join(input_trajectories_path, "input.txt"), "w") as file:
                for line in self.tokenized_trajectories:
                    file.write(line + "\n")

            try:
                result = subprocess.run(
                    ["python", os.path.join(input_trajectories_path, "prepare.py")],
                    capture_output=True,
                    text=True,
                )
                print("Script output:", result.stdout)
                print("Data prepared successfully for the model...")
            except subprocess.CalledProcessError as e:
                print("Error preparing data for the model:", e)
            print("Starting Model Training Now...")
            # Assuming I have the new model architecure
            os.chdir(working_directory)
            try:
                process = subprocess.run(
                    ["python", training_model_path, configurations_model_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print("Script output:", process.stdout)
                print("Trained model for generation successfully.")
            except subprocess.CalledProcessError as e:
                print("Error training model for generation:", e)
                print("Script error output:", e.stderr)

        elif self.mode == "summarization_testing":
            # I need to pass the given self.tokenized_trajectories to /speakingTrajectories/Transformers/nanoGPT/requestedTrajectories.txt
            # Then run ./generateTrajectoriesScript.sh 1 2 and start the summarization process with the (spatial constrainsts?).
            requested_trajectories_path = os.path.join(
                transformers_path, "nanoGPT/requestedTrajectories.txt"
            )
            simplified_trajectories_path = os.path.join(
                transformers_path, "nanoGPT/simplifiedTrajectories.txt"
            )
            with open(requested_trajectories_path, "w") as file:
                for line in self.tokenized_trajectories:
                    file.write(line + "\n")

            script_path = os.path.join(
                transformers_path, "generateTrajectoriesScript.sh"
            )
            arg1 = 2
            # arg2 = 1
            # arg3 = 1

            try:
                result = subprocess.run(
                    [script_path, str(arg1)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print("Script output:", result.stdout.decode())
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e)
                print("Script error output:", e.stderr.decode())

            # Open the source file in read mode and the destination file in write mode
            with open(simplified_trajectories_path, "r") as source_file:
                with open(final_trajectories_path, "w") as destination_file:
                    # Read content from the source file
                    content = source_file.read()
                    # Write content to the destination file
                    destination_file.write(content)
        elif self.mode == "generation_testing":
            print("Started generating trajectories...")
            generated_trajectories_path = os.path.join(
                transformers_path, "nanoGPT/generatedTrajectories.txt"
            )
            script_path = os.path.join(
                transformers_path, "generateTrajectoriesScript.sh"
            )
            arg1 = 1
            arg2 = self.trajectories_count
            arg3 = self.trajectories_length

            try:
                result = subprocess.run(
                    [script_path, str(arg1), str(arg2), str(arg3)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print("Script output:", result.stdout.decode())
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script:", e)
                print("Script error output:", e.stderr.decode())

            # Open the source file in read mode and the destination file in write mode
            with open(generated_trajectories_path, "r") as source_file:
                with open(final_trajectories_path, "w") as destination_file:
                    # Read content from the source file
                    content = source_file.read()
                    # Write content to the destination file
                    destination_file.write(content)
        pass

    def summarize_trajectories(self):
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
        # Call your ML model for summarization
        generated_summaries = ["Generated summary for trajectory"] * len(
            self.trajectories
        )
        return generated_summaries

    def generate_trajectories(self):
        if not self.params:
            raise ValueError("No parameters loaded")
        # Call your ML model for trajectory generation
        generated_trajectories = ["Generated trajectory"] * self.params[
            "number_of_trajectories"
        ]
        return generated_trajectories

    def run_pipeline(
        self,
        output_filepath: str = "output.json",
    ):
        if self.mode == "summarization_training":

            # Tokenization
            self.tokenizationModule()
            # Finetuning
            self.fineTuningModule()
            # Spatial Constrains
            self.spatialConstraintsModule()
            # Proceed with training your summarization model using self.trajectories and self.summaries
            # Save the model in the models repo
            self.modelsRepository()
            print("New model saved to repository...")

        elif self.mode == "generation_training":

            # Tokenization
            self.tokenizationModule()
            # Finetuning
            self.fineTuningModule()
            # Spatial Constrains
            self.spatialConstraintsModule()
            # Proceed with training your generation model using self.trajectories
            # Save the model in the models repo
            self.modelsRepository()
            print("New model saved to repository...")

        elif self.mode == "summarization_testing":
            self.tokenizationModule()
            # Call the suitable model in the models repo and summarize trajectories
            self.modelsRepository()
            # summaries = self.summarize_trajectories()
            # Apply the spatial constraints on the genrated summarizes or make sure they are applied...think later about this
            self.spatialConstraintsModule()
            # Apply the detokenization module and save the output
            self.deTokenizationModule()
            # output_data = [
            #     {"trajectory": traj, "summary": summary}
            #     for traj, summary in zip(self.trajectories, summaries)
            # ]
            # self.save_data(output_filepath, output_data)

        elif self.mode == "generation_testing":
            print("I am doing Trajectory Generation Testing from the Pipeline now")
            # Call the suitable model in the models repo and generate trajectories
            self.modelsRepository()
            # trajectories = self.generate_trajectories()
            # Apply the spatial constraints on the genrated output or make sure they are applied...think later about this
            self.spatialConstraintsModule()
            # Apply the detokenization module and save the output
            self.deTokenizationModule()
            # output_data = [{"trajectory": traj} for traj in trajectories]
            # self.save_data(output_filepath, output_data)

        else:
            raise ValueError(
                "Invalid mode. Choose from 'summarization_training', 'generation_training', 'summarization_testing', or 'generation_testing'"
            )


# Example usage
# pipeline = TrajectoryPipeline()
# pipeline.load_params("/speakingTrajectories/Pipeline/Input/params.json")
# pipeline.run_pipeline(
#     output_filepath="summarized_trajectories.json",
# )
