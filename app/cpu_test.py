import torch
print(torch.version.cuda)        # Should say something like '11.8' or '12.1'
print(torch.cuda.is_available()) # Should be True
print(torch.cuda.get_device_name(0))  # Should print your GPU name
