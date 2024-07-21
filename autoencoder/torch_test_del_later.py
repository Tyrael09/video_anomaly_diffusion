import torch

def check_gpu():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Script is using GPU: {torch.cuda.get_device_name(device)} (ID: {device})")
    else:
        print("CUDA is not available. Running on CPU.")

if __name__ == "__main__":
    check_gpu()
