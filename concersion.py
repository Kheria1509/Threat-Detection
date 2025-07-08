import torch

# Load your PyTorch model
model_paths = [
    'detect/train/weights/best.pt',
    'detect/weapondetction1_train/weights/best.pt',
    'detect/weapondetction1_train/weights/best.pt',
    'detect/fire_smoke_train/weights/best.pt'
]

for i, path in enumerate(model_paths):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Load the checkpoint dictionary
    model = checkpoint['model'].float()  # Extract the model from the checkpoint and convert to float
    model.eval()  # Set the model to evaluation mode

    # If your model is wrapped in DataParallel, unwrap it
    if 'DataParallel' in str(type(model)):
        model = model.module

    # Create dummy input (replace this with your actual input shape)
    dummy_input = torch.randn(1, 3, 224, 224).float()  # Convert dummy input to float

    # Export the model to ONNX
    input_names = ["input"]
    output_names = ["output"]
    onnx_path = f"model_{i}.onnx"  # Output ONNX file path
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names)

    print(f"Model {i + 1} converted to ONNX successfully and saved as {onnx_path}")
