import torch
from autoencoder.feat_models import GCLModel  # Adjust the import based on your actual model location

# TODO NO NEED TO USE THIS!!


# Define the function to load the model
def load_model(model_path, model_config):
    model = K.models.GVADModel(model_config['feat_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Example usage
model_path = "checkpoints/model_epoch_90.pth"  # Adjust the path to your saved model
model_config = {"feat_size": feat_size}  # Replace with your actual model configuration

# Load the model
loaded_model = load_model(model_path, model_config)

# Example input tensor for inference
example_input = torch.randn(1, model_config['feat_size'])  # Adjust the shape as necessary

# Run inference
with torch.no_grad():
    output = loaded_model(example_input)

print(output)
