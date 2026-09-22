import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        
        self.weight_left = nn.Parameter(kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, original, enhanced):
        original_mean = torch.mean(original, 1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, 1, keepdim=True)

        original_pool = self.pool(original_mean)
        enhanced_pool = self.pool(enhanced_mean)

        D_org_left = F.conv2d(original_pool, self.weight_left.to(original.device), padding=1)
        D_org_right = F.conv2d(original_pool, self.weight_right.to(original.device), padding=1)
        D_org_up = F.conv2d(original_pool, self.weight_up.to(original.device), padding=1)
        D_org_down = F.conv2d(original_pool, self.weight_down.to(original.device), padding=1)

        D_enh_left = F.conv2d(enhanced_pool, self.weight_left.to(enhanced.device), padding=1)
        D_enh_right = F.conv2d(enhanced_pool, self.weight_right.to(enhanced.device), padding=1)
        D_enh_up = F.conv2d(enhanced_pool, self.weight_up.to(enhanced.device), padding=1)
        D_enh_down = F.conv2d(enhanced_pool, self.weight_down.to(enhanced.device), padding=1)

        D_left = torch.pow(D_org_left - D_enh_left, 2)
        D_right = torch.pow(D_org_right - D_enh_right, 2)
        D_up = torch.pow(D_org_up - D_enh_up, 2)
        D_down = torch.pow(D_org_down - D_enh_down, 2)

        return torch.mean(D_left + D_right + D_up + D_down)

class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2))

class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        D_rg = torch.pow(mr - mg, 2)
        D_rb = torch.pow(mr - mb, 2)
        D_gb = torch.pow(mb - mg, 2)
        return torch.mean(torch.pow(torch.pow(D_rg, 2) + torch.pow(D_rb, 2) + torch.pow(D_gb, 2), 0.5))

class ZeroIDIRLosses(nn.Module):
    def __init__(self):
        super(ZeroIDIRLosses, self).__init__()
        self.l_spa = SpatialConsistencyLoss()
        self.l_exp = ExposureControlLoss(patch_size=16, mean_val=0.6)
        self.l_col = ColorConstancyLoss()
        self.l_tv = IlluminationSmoothnessLoss()
        
        # Weights for Stage 1 (Illumination Correction)
        self.w_spa = 1.0
        self.w_exp = 10.0
        self.w_col = 5.0
        self.w_tv = 200.0

    def compute_stage1_loss(self, enhanced_image, original_image, illumination_map):
        loss_spa = self.l_spa(original_image, enhanced_image)
        loss_exp = self.l_exp(enhanced_image)
        loss_col = self.l_col(enhanced_image)
        loss_tv = self.l_tv(illumination_map)
        
        total_stage1_loss = (
            self.w_spa * loss_spa + 
            self.w_exp * loss_exp + 
            self.w_col * loss_col + 
            self.w_tv * loss_tv
        )
        return total_stage1_loss, loss_spa, loss_exp, loss_col, loss_tv

    def compute_diffusion_consistency_loss(self, noise_pred, noise_target, x_start_pred, x_start_target):
        # MSE Loss for the diffusion noise prediction
        noise_loss = F.mse_loss(noise_pred, noise_target)
        
        # Consistency loss pushing the diffusion to align with the pseudo-clean stage 1 image
        x0_loss = F.mse_loss(x_start_pred, x_start_target)
        
        total_diffusion_loss = noise_loss + x0_loss
        return total_diffusion_loss
