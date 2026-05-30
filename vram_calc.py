import torch

if torch.cuda.is_available():
    # Get total VRAM in bytes, then convert to Gigabytes (GB)
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory_bytes / 1e9

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_memory_gb:.2f} GB")

    # Rule of thumb: fp16 uses 2 bytes per parameter.
    # Therefore, Number of Parameters = Total Bytes / 2
    max_parameters = total_memory_bytes / 2

    # Convert to billions (B) for easier reading
    max_parameters_billion = max_parameters / 1e9

    print("---")
    print("Rule of Thumb (fp16): 2 bytes per parameter")
    print(
        f"Absolute Maximum Model Size: {max_parameters_billion:.2f} Billion Parameters"
    )
    print(
        "*(Note: In reality, you need ~50% of VRAM for gradients and optimizer states during training, so your practical limit is half of this!)*"
    )
else:
    print("CUDA is not available.")
