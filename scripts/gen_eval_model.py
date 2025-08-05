import torch
from pytorch_msssim import MS_SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

dummy_input1_lpips = torch.rand(1, 3, 480, 640).cuda()
dummy_input2_lpips = torch.rand(1, 3, 480, 640).cuda()

dummy_input1_ssim = torch.rand(1, 3, 480, 640).cuda()
dummy_input2_ssim = torch.rand(1, 3, 480, 640).cuda()

scripted_lpips = torch.jit.trace(lpips, (dummy_input1_lpips, dummy_input2_lpips))

scripted_lpips.save("./scripts/lpips_model.pt")

ms_ssim = ms_ssim.cuda()

scripted_ms_ssim = torch.jit.trace(ms_ssim, (dummy_input1_ssim, dummy_input2_ssim))

scripted_ms_ssim.save("./scripts/ms_ssim_model.pt")
