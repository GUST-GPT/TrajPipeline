"""Main entry point for the user to the pipeline"""

# Generate Full UML Diagram pyreverse -o png -p all *.py
from Pipeline import TrajectoryPipeline

my_pipeline = TrajectoryPipeline(
    mode="training",
    use_tokenization=True,
    use_detokenization=True,
    use_spatial_constraints=True,
    modify_spatial_constraints=True,
    use_predefined_spatial_constraints=True,
)

# Provide trajectories if tokenization is enabled
trajectories = [
    [(37.7749, -122.4194), (37.7750, -122.4195)],
    [(34.0522, -118.2437), (34.0523, -118.2438)],
]


# User creates their own rule dynamically
def user_defined_rule(token, previous_tokens):
    """Example of a user defined rule to be added
    to the spatial constraints module"""
    return len(previous_tokens) < 10  # Example condition


# User defines the rules they want to apply
rules = [user_defined_rule]

my_pipeline.set_trajectories(trajectories)
my_pipeline.set_tokenization_resolution(10)
my_pipeline.define_spatial_constraints(rules=rules)

# Process the pipeline
my_pipeline.run()
