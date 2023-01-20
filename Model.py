import torch
import torch.nn as nn
from torch.nn.init import normal_

class Registry:
    def __init__(self):
        self.funcs = dict()
    
    def register(self, func):
        self.funcs[func.__name__] = func
        return func
    
    def get_func(self, func_name):
        return self.funcs.get(func_name, None)

conv_manager = Registry()
att_manager = Registry()

class DC_layer(nn.Module):
    def __init__(self, kdata_input=True):
        super(DC_layer, self).__init__()
        self.kdata_input = kdata_input

    def forward(self, mask, x_rec, x_under):
        if self.kdata_input:
            x_rec_per = x_rec.permute(0, 2, 3, 1)
            x_rec_per = torch.complex(x_rec_per[:, :, :, 0], x_rec_per[:, :, :, 1])
            x_tran_per = torch.fft.fftn(x_rec_per, dim=(1, 2))
            x_tran_per_real =  torch.unsqueeze((x_tran_per.real), -1)
            x_tran_per_imag =  torch.unsqueeze((x_tran_per.imag), -1)
            concat = [x_tran_per_real, x_tran_per_imag]
            x_tran_per = torch.cat(concat, dim=-1)
            x_tran = x_tran_per.permute(0, 3, 1, 2)
            masknot = 1 - mask
            output = masknot * x_tran + x_under
            output_per = output.permute(0, 2, 3, 1)
            output_per = torch.complex(output_per[:, :, :, 0], output_per[:, :, :, 1])
            output_tran_per = torch.fft.ifftn(output_per, dim=(1, 2))
            output_tran_per_real =  torch.unsqueeze((output_tran_per.real), -1)
            output_tran_per_imag =  torch.unsqueeze((output_tran_per.imag), -1)
            concat = [output_tran_per_real, output_tran_per_imag]
            output_tran_per = torch.cat(concat, dim=-1)
            final_recon = output_tran_per.permute(0, 3, 1, 2)

        else:
            recon_k = torch.fft.fftshift(torch.fft.fft2(x_rec))
            under_k = torch.fft.fftshift(torch.fft.fft2(x_under))
            masknot = 1 - mask
            output = masknot * recon_k + under_k * mask
            final_recon = torch.abs(torch.fft.ifft2(torch.fft.fftshift(output)))
            
        return final_recon

@att_manager.register
class ECA(nn.Module):
    def __init__(self, channel, k_size=3, **kwargs):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2,dilation=1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

@att_manager.register
class SE(nn.Module):
    def __init__(self, in_channel, r=8, **kwargs):
        super(SE, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s*x

@att_manager.register
class CSA(nn.Module):
    def __init__(self, in_channel, conv_num=2, **kwargs):
        super(CSA, self).__init__()
        layers = [nn.AdaptiveAvgPool2d(1)]
        for i in range(conv_num-1):
            layers = layers + [nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=in_channel), nn.ReLU()]
        layers = layers + [nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=in_channel), nn.Sigmoid()]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        s = self.layer(x)
        return s * x

@att_manager.register
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7, **kwargs):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

@att_manager.register
class CSABAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7, **kwargs):
        super(CSABAM, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1, groups=channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, groups=channel),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_out = self.layer(x)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

@conv_manager.register
class DoubleConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, shape, bottole_neck=False, attention_type="none", **kwargs):
        super(DoubleConv, self).__init__()
        if bottole_neck:
            self.doubleConv = nn.Sequential(
                nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(inchannels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannels*2, inchannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inchannels),
                nn.ReLU(inplace=True)
            )
        else:
            self.doubleConv = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannels),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannels),
                nn.ReLU(inplace=True)
            )
        self.att_func = att_manager.get_func(attention_type)
        if not self.att_func is None:
            self.att_layer = self.att_func(outchannels, conv_num=kwargs["csa_k"])

    def forward(self, input):
        out = self.doubleConv(input)
        if not self.att_func is None:
            out = self.att_layer(out)
        return out

@conv_manager.register
class ResDoubleConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, shape, bottole_neck=False, attention_type="none", **kwargs):
        super(ResDoubleConv, self).__init__()
        if bottole_neck:
            self.conv1 = nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(inchannels*2)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(inchannels*2, inchannels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(inchannels)
        else:
            self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(outchannels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(outchannels)
        self.stride = stride
        if stride == 2 or inchannels != outchannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(outchannels),
            )
        else:
            self.downsample = None
        self.att_func = att_manager.get_func(attention_type)
        if not self.att_func is None:
            self.att_layer = self.att_func(outchannels, shape=shape, conv_num=kwargs["csa_k"])
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if not self.att_func is None:
            out = self.att_layer(out)

        if not self.downsample is None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class UpSample(nn.Module):
    def __init__(self, factor=2):
        super(UpSample,self).__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self,input):
        return self.up(input)

class Unet(nn.Module):
    def __init__(self, inchannels, outchannels, ndf, conv_type, att_type, csa_k):
        super(Unet, self).__init__()
        conv_func = conv_manager.get_func(conv_type)

        self.enConv1 = DoubleConv(inchannels, ndf, stride=1, shape=(256, 256))
        self.enConv2 = conv_func(ndf, ndf*2, stride=2, shape=(128, 128), attention_type=att_type, csa_k=csa_k)
        self.enConv3 = conv_func(ndf*2, ndf*4, stride=2, shape=(64, 64), attention_type=att_type, csa_k=csa_k)
        self.bottoleNeck = conv_func(ndf*4, ndf*4, stride=2, shape=(32, 32), bottole_neck=True, attention_type=att_type, csa_k=csa_k)

        self.deConv3 = conv_func(ndf*8, ndf*2, stride=1, shape=(64, 64), attention_type=att_type, csa_k=csa_k)
        self.deConv2 = conv_func(ndf*4, ndf*1, stride=1, shape=(128, 128), attention_type=att_type, csa_k=csa_k)
        self.deConv1 = conv_func(ndf*2, ndf*1, stride=1, shape=(256, 256), attention_type=att_type, csa_k=csa_k)

        self.up1 = UpSample(2)
        self.up2 = UpSample(2)
        self.up3 = UpSample(2)

        self.outConv = nn.Conv2d(ndf*1, outchannels, kernel_size=1)
        self.dc = DC_layer(inchannels==2)

    def forward(self, input, mask):
        en1 = self.enConv1(input)
        en2 = self.enConv2(en1)
        en3 = self.enConv3(en2)
        bk = self.bottoleNeck(en3)

        de_up3 = self.up3(bk)
        cat3 = torch.cat([de_up3, en3], dim=1)
        de3 = self.deConv3(cat3)

        de_up2 = self.up2(de3)
        cat2 = torch.cat([de_up2, en2], dim=1)
        de2 = self.deConv2(cat2)

        de_up1 = self.up1(de2)

        cat1 =  torch.cat([de_up1, en1], dim=1)
        de1 = self.deConv1(cat1)

        out = self.outConv(de1)
        out = torch.clamp(out, 0, 1)

        x_rec_dc = self.dc(mask, out, input)
        return x_rec_dc

class meta1_UNetCSE(nn.Module):
    def __init__(self, inchannels, outchannels, ndf, conv_type, att_type, csa_k):
        super(meta1_UNetCSE, self).__init__()   
        conv_func = conv_manager.get_func(conv_type)
        self.ndf = ndf
        self.inc = DoubleConv(inchannels, self.ndf, stride=1, shape=(256, 256))
        self.down1 = nn.Sequential(nn.MaxPool2d(2),
                                   DoubleConv(self.ndf, self.ndf*2, stride=1, shape=(128, 128)))
        
        self.down2 = nn.Sequential(nn.MaxPool2d(2),
                                   conv_func(self.ndf*2, self.ndf*4, stride=1, shape=(64, 64), attention_type=att_type, csa_k=csa_k))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.ndf*4, self.ndf*2, kernel_size=3, padding=1))
        self.conv2 = conv_func(self.ndf*4, self.ndf*2, stride=1, shape=(128, 128), attention_type=att_type, csa_k=csa_k)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.ndf*2, self.ndf, kernel_size=3, padding=1))
        
        self.conv1 = conv_func(self.ndf*2, self.ndf, stride=1, shape=(256, 256), attention_type=att_type, csa_k=csa_k)
        
        self.outc = nn.Conv2d(self.ndf, outchannels, 1)

    def forward(self, x):
        en1 = self.inc(x)
        en2 = self.down1(en1)
        en3 = self.down2(en2)
        de_up2 = self.up2(en3)
        cat2 = torch.cat([de_up2, en2], dim=1)
        de2 = self.conv2(cat2)
        de_up1 = self.up1(de2)
        cat1 =  torch.cat([de_up1, en1], dim=1)
        de1 = self.conv1(cat1)
        out = self.outc(de1)
        return out

class MICCANlong1(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, ndf, conv_type, att_type, csa_k):
        super(MICCANlong1, self).__init__()
        self.layer = nn.ModuleList([meta1_UNetCSE(in_channel, out_channel, ndf, conv_type, att_type, csa_k) for _ in range(n_layer)])
        self.dc = DC_layer(in_channel==2)
        self.nlayer = n_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_zf, mask):  
        x_rec = x_zf
        for i in range(self.nlayer - 1):
            x_rec_ = self.layer[i](x_rec)
            x_rec = x_rec_ + x_rec
            x_rec = self.dc(mask, x_rec, x_zf)

        x_rec = self.layer[i+1](x_rec)
        x_rec = x_zf + x_rec
        x_rec = self.dc(mask, x_rec, x_zf)
        x_rec = torch.clamp(x_rec, 0, 1)
        return x_rec

class MICCANlong2(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, ndf, conv_type, att_type, csa_k):
        super(MICCANlong2, self).__init__()
        self.layer = nn.ModuleList([meta1_UNetCSE(in_channel, out_channel, ndf, conv_type, att_type, csa_k) for _ in range(n_layer)])
        self.dc = DC_layer(in_channel==2)
        self.nlayer = n_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_under, mask):
        x_under_per = x_under.permute(0, 2, 3, 1)
        x_under_per = torch.complex(x_under_per[:, :, :, 0], x_under_per[:, :, :, 1])
        x_zf_per = torch.fft.ifftn(x_under_per,dim=(1, 2))
        x_zf_per_real =  torch.unsqueeze((x_zf_per.real), -1)
        x_zf_per_imag =  torch.unsqueeze((x_zf_per.imag), -1)
        concat = [x_zf_per_real, x_zf_per_imag]
        x_zf_per = torch.cat(concat, dim=-1)

        x_zf = x_zf_per.permute(0, 3, 1, 2)
        x_rec_dc = x_zf
        recimg = list()
        recimg.append(self.sigtoimage(x_zf)) 
  
        for i in range(self.nlayer - 1):
            x_rec = self.layer[i](x_rec_dc)
            x_res = x_rec_dc + x_rec
            x_rec_dc = self.dc(mask, x_res, x_under)
            recimg.append(self.sigtoimage(x_rec_dc))
        x_rec = self.layer[i+1](x_rec_dc)
        x_res = x_zf + x_rec
        x_rec_dc = self.dc(mask, x_res, x_under)
        recimg.append((self.sigtoimage(x_rec_dc)))
        return recimg[-1]

    def sigtoimage(self, sig):
        x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
        x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
        x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
        return x_image