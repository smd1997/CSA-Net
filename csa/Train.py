# -*- coding: utf-8 -*-
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from types import SimpleNamespace
from Model import Unet, MICCANlong1, MICCANlong2
from MyDataset import MyDataSet
from myEarlystop import EarlyStopping
from eval import eval_net
from torch.utils.tensorboard import SummaryWriter
import os
from loss import Percetual

def train(args):
    logging.info('****** Loading model ******')
    if args.save_name.split('_')[0] == "miccanlong1":
        model = MICCANlong1(args.inchannel, args.outchannel, 5, args.ndf, args.convType, args.attType, args.csa_k)
    elif args.save_name.split('_')[0] == "miccanlong2":
        model = MICCANlong2(args.inchannel, args.outchannel, 5, args.ndf, args.convType, args.attType, args.csa_k)
    elif args.save_name.split('_')[0] == "unet":
        model = Unet(args.inchannel, args.outchannel, args.ndf, args.convType, args.attType, args.csa_k)

    model = model.to(args.device)
    logging.info(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lrReducePatience, factor=0.5) 

    logging.info('loading train dataset......')
    trainData = MyDataSet(args.maskName, args.dataDir, args.trainMat, kdata_input=args.inchannel==2, mode="train", transform=True)
    logging.info('loading val dataset......')
    valData = MyDataSet(args.maskName, args.dataDir, args.valMat, kdata_input=args.inchannel==2, mode="val", transform=False)
    logging.info('loading test dataset......')
    testData = MyDataSet(args.maskName, args.dataDir, args.testMat, kdata_input=args.inchannel==2, mode="test", transform=False)

    train_loader = DataLoader(trainData, batch_size = args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valData, batch_size = 1, shuffle = False, num_workers= args.val_num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(testData, batch_size = 1, shuffle = False, num_workers= args.val_num_workers, pin_memory=True, drop_last=False)
    early_stopping = EarlyStopping(patience=args.stopPatience, savepath=os.path.join(args.output_dir, args.save_name), cpname="CP_early_stop", verbose=True)

    #Init tensor board writer
    if args.tb_write:  
        writer = SummaryWriter(log_dir=os.path.join(args.tb_dir, f'tensorboard_{args.save_name}'))

    #Init the loss
    criterion = {"combo":Percetual()}.get(args.loss, torch.nn.MSELoss())
    min_val_loss = 100000.0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
        Loss:            {criterion}
    ''')
    
    try:
        for epoch in range(0, args.epochs):
            model.train()
            progress = 0
            train_tot_loss = [0, 0]
            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs}', unit=' imgs') as pbar:
                for step, (label_image, fold_image, un_mask) in enumerate(train_loader):
                    label_image = label_image.to(device = args.device, dtype = torch.float32)
                    fold_image = fold_image.to(device = args.device, dtype = torch.float32)                    
                    un_mask = un_mask.to(device = args.device, dtype = torch.float32)
                    recon_img = model(fold_image, un_mask)
                    
                    train_loss = criterion(recon_img, label_image)
                    optimizer.zero_grad()
                    # import pdb;pdb.set_trace()
                    train_loss.backward()
                    optimizer.step()

                    progress += 100*args.batch_size/len(trainData)
                    pbar.set_postfix(**{ 'loss': train_loss.item(), 'Prctg of train set': progress})
                    pbar.update(args.batch_size)
                    train_tot_loss = [train_tot_loss[0] + train_loss.item(), train_tot_loss[1] + 1]
                writer.add_scalar('train/Loss', train_tot_loss[0] / train_tot_loss[1], epoch)
        
            logging.info("Learning rate: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            val_recon_img, val_label_img, val_loss, val_psnr, val_ssim = eval_net(model, val_loader, criterion, args)
            logging.info('Epoch: {}, validation full score, ImL2: {}, PSNR: {}, SSIM: {}'.format(epoch+1, val_loss, val_psnr, val_ssim))
            
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

            if epoch > args.earlyStopEpoch:
                scheduler.step(val_loss)
                early_stopping(val_loss, model)
                if early_stopping.early_stop == True:
                    print("Early stopping!")
                    break

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.output_dir, f'{args.save_name}', 'Best_CP_epoch.pth'))

                logging.info('Best Checkpoint saved !')

            if (epoch % args.save_step) == (args.save_step-1):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(args.output_dir, f'{args.save_name}', f'CP_epoch{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(args.output_dir, f'{args.save_name}', f'CP_epoch{epoch + 1}_INTERRUPTED.pth'))
        logging.info('Saved interrupt')
    best_state_dict = torch.load(os.path.join(args.output_dir, f'{args.save_name}', 'Best_CP_epoch.pth'))
    model.load_state_dict(best_state_dict['model_state_dict'])
    *_, best_loss, best_psnr, best_ssim = eval_net(model, test_loader, criterion, args)
    logging.info('Test result for best epoch {}, ImL2: {}, PSNR: {}, SSIM: {}'.format(best_state_dict["epoch"], best_loss, best_psnr, best_ssim))
    writer.close()

def get_args():
    ## get all the paras from config.yaml
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    return args

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
     # Create output dir
    os.makedirs(os.path.join(args.output_dir, args.save_name), exist_ok=True)
    logging.info('Created checkpoint directory')
    
    # Copy configuration file to output directory
    with open(os.path.join(args.output_dir, args.save_name, 'config.yaml'), "w") as f:
        yaml.dump(vars(args), f)
    logging.info(f'Using device {args.device}')
    train(args)

if __name__ == "__main__":
    main()
