import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

model = torch.nn.Linear(10, 1).to(device)
ddp_model = DDP(model, device_ids=[local_rank])
dummy_data = torch.randn(8, 10).to(device)
dummy_target = torch.randn(8, 1).to(device)
optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

optimizer.zero_grad()
output = ddp_model(dummy_data)
loss = torch.nn.functional.mse_loss(output, dummy_target)
loss.backward()
optimizer.step()

print(f"Rank {dist.get_rank()} on device {device} completed one step successfully.", flush=True)

dist.destroy_process_group()