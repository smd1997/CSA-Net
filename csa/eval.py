import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def eval_net(net, loader, Loss1, args):
    net.eval()
    tot_val_loss = 0
    psnr_list = []
    ssim_list = [] 

    for slice_num, (label_image, fold_image, un_mask) in enumerate(tqdm(loader)):

        fold_image = fold_image.to(device=args.device, dtype=torch.float32)
        label_image = label_image.to(device=args.device, dtype=torch.float32)
        un_mask = un_mask.to(device=args.device, dtype=torch.float32)

        with torch.no_grad():
            recon_image = net(fold_image, un_mask)
        
        val_loss = Loss1(recon_image, label_image)
        tot_val_loss = tot_val_loss + val_loss.item()

        recon_image_ = np.squeeze(recon_image.cpu().numpy())
        label_image_ = np.squeeze(label_image.cpu().numpy())
        rec_psnr = psnr(label_image_, recon_image_)
        rec_ssim = ssim(recon_image_, label_image_)
       
        psnr_list.append(rec_psnr)
        ssim_list.append(rec_ssim)

    return recon_image, label_image, tot_val_loss / len(loader), np.mean(psnr_list), np.mean(ssim_list)
    