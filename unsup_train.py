import argparse
from pathlib import Path
import time
from turtle import forward
from dataset import *
from models import Unsupervised
from torch.utils.tensorboard import SummaryWriter
import warnings
import torchvision
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(seed=1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

PRINT_INTERVAL = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = 'cpu'

class AverageMeter(object):

    def __init__(self, keep_all=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def epoch(model, data, criterion, epoch_index,optimizer=None):
    np.random.seed(seed=1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_smooth_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    avg_epe = AverageMeter()

    tic = time.time()
    imgs0,flow0 = next(iter(data))
    imgs0 = imgs0.to(device)[:8]
    flow0 = flow0.to(device)[:8]
    for i, (imgs, flow) in enumerate(data):
        imgs = imgs.to(device)
        flow = flow.to(device)
        # imgs = imgs0
        # flow = flow0
        #print(flow.sum().item()*0,time.time() - tic, "one")
        #tic = time.time()
        with torch.set_grad_enabled(optimizer is not None):
            pred_flows, wraped_imgs, masks, depth_metrics = model(imgs,flow,return_depth=True)
            #print(pred_flows[0].sum().item()*0,time.time() - tic, "two")
            #tic = time.time()
            loss, bce_loss, smooth_loss = criterion(pred_flows, wraped_imgs, masks, imgs[:, :3, :, :], imgs[:, 3:, :, :])
            #print(loss.item()*0,time.time() - tic, "three")
            #tic = time.time()

        if optimizer is not None:
            for j in range(5):
                # def print_grad(x):
                #     print(f"pred_flows {x.shape} {x.abs().max():.3f}")
                # pred_flows[j].register_hook(print_grad)
                pred_flows[j].retain_grad()
                pred_flows[j].requires_grad_(True)
        #pred_flow = F.interpolate(pred_flows[0], (flow.shape[2], flow.shape[3]), mode='bilinear', align_corners=True)
        pred_flow = pred_flows[0]
        flow = F.interpolate(flow, (pred_flow.shape[2], pred_flow.shape[3]), mode='bilinear', align_corners=True) * pred_flow.shape[2]/flow.shape[2]
        epe = torch.sum((flow.to(device)-pred_flow)**2,dim=1).sqrt().mean()

        #print(epe.item()*0,time.time() - tic, "four")
        #tic = time.time()
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            print("Model grad")
            for name,param in model.named_parameters():
                if param.grad is None:
                    print(name,"no grad")
                else:
                    print(name,param.grad.min(),param.grad.max(),param.grad.mean(),param.grad.abs().mean())
            print("max flow:")
            for j in range(5):
                print(f"{pred_flows[j].abs().max():.3f}",end=" ")
            print()
            print("mean flow:")
            for j in range(5):
                print(f"{pred_flows[j].mean():.3f}",end=" ")
            print()
            print("mean grad flow:")
            for j in range(5):
                if pred_flows[j].grad is not None:
                    print(f"{pred_flows[j].grad.mean():.3f}",end=" ")
            print()
            print("max grad flow:")
            for j in range(5):
                if pred_flows[j].grad is None:
                    continue
                print(f"{pred_flows[j].grad.abs().max():.3f}",end=" ")
                
                
                big_grads = (pred_flows[j].grad.abs()>2).float().sum()
                if big_grads>=1:
                    print()
                    print("BIG GRADIENTS: ",big_grads.item())
                    
                    torch.save(pred_flows[j],"tensors/pred_flows")
                    torch.save(wraped_imgs[j],"tensors/warped")
                    torch.save(imgs,"tensors/imgs")
                    torch.save(depth_metrics[j],"tensors/depth_metric")
                    torch.save(masks[j],"tensors/mask")
                    exit()
                #print(f"{wraped_imgs[j].grad.abs().max():.3f}",end=" ")
            print()
            optimizer.step()
            
        #print(model.predictor.conv1[0].weight.grad.sum().item()*0,time.time() - tic, "five")
        #tic = time.time()

        batch_time = time.time() - tic
        tic = time.time()
        avg_epe.update(epe.item())
        avg_bce_loss.update(bce_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)
        
        print("",flush=True)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
                  'bce_loss {bce.val:5.4f} ({bce.avg:5.4f})\t'
                  'epe {epe.val:5.4f} ({epe.avg:5.4f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                smooth=avg_smooth_loss, bce=avg_bce_loss, epe=avg_epe),flush=True)
            #print(model.predictor.upconvflow3.weight.sum().item(),flush=True)
        
        if i%50 == 0:
            path = f"images/epoch{epoch_index}_{i}_{model.forward_splat}"
            Path(path).mkdir(exist_ok=True)
            pyramid_ind = 0
            img1 = F.interpolate(imgs[:,:3], (pred_flows[pyramid_ind].shape[2], pred_flows[pyramid_ind].shape[3]), mode='bilinear', align_corners=True)[0]
            img2 = F.interpolate(imgs[:,3:], (pred_flows[pyramid_ind].shape[2], pred_flows[pyramid_ind].shape[3]), mode='bilinear', align_corners=True)[0]
            torchvision.utils.save_image(img1,f"{path}/img1.png")
            torchvision.utils.save_image(img2,f"{path}/img2.png")
            torchvision.utils.save_image(wraped_imgs[pyramid_ind][0],f"{path}/warped_{epoch_index}_{i*0}.png")
            #torchvision.utils.save_image(model.stn_splat(flow[:1],img1.unsqueeze(0), torch.ones_like(flow)[:,:1])[0][0],f"{path}/warped_gt.png")
            computeImg(pred_flows[pyramid_ind][0].cpu().detach().numpy(), verbose=True, savePath=f'{path}/predicted_flow_{epoch_index}_{i*0}.png')
            computeImg(flow[0].cpu().detach().numpy(), verbose=True, savePath=f'{path}/true_flow.png')

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg bce_loss {bce.avg:5.4f} \t'
          'Avg epe {epe.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, bce=avg_bce_loss,epe=avg_epe),flush=True)

    return avg_smooth_loss.avg, avg_bce_loss.avg, avg_loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../FlyingChairs_release/data', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', default='flownet', type=str, help='the supervised model to be trained with ('
                                                                     'flownet, lightflownet, pwc_net)')
    parser.add_argument('--steps', default=6000000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=5e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")
    parser.add_argument("--forward_splat", help="Use forward splatting instead of backward warp", action="store_true")
    parser.add_argument("--transfer", help="perform transfer learning from an already trained supervised model",
                        action="store_true")

    args = parser.parse_args()

    mymodel = Unsupervised(conv_predictor=args.model, forward_splat=args.forward_splat)
    mymodel.to(device)
    path = os.path.join("Unsupervised", type(mymodel.predictor).__name__)
    if args.forward_splat:
        loss_fnc = unsup_loss_forward_warp
    else:
        loss_fnc = unsup_loss_backwarp
    if args.transfer:
        best_model = torch.load(os.path.join("model_weight", type(mymodel.predictor).__name__, 'best_weight.pt'),
                                map_location=device)
        mymodel.predictor.load_state_dict(best_model['model_state_dict'])

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)

    co_aug_transforms = None
    frames_aug_transforms = None

    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensor()
    ])

    if args.augment:
        if "Chairs" in args.root:
            crop = albu.RandomSizedCrop((150, 384), 384, 512, w2h_ratio=512 / 384, p=0.5)
        elif "sintel" in args.root:
            crop = albu.RandomSizedCrop((200, 436), 436, 1024, w2h_ratio=1024 / 436, p=0.5)
        else:
            crop = albu.RandomSizedCrop((200, 400), 250, 250, w2h_ratio=1, p=0.5)
        co_aug_transforms = albu.Compose([
            crop,
            albu.Flip(),
            albu.ShiftScaleRotate()
        ])

        frames_aug_transforms = albu.Compose([
            albu.OneOf([albu.Blur(), albu.MedianBlur(), albu.MotionBlur()], p=0.5),

            albu.OneOf([albu.OneOf([albu.HueSaturationValue(), albu.RandomBrightnessContrast()], p=1),

                        albu.OneOf([albu.CLAHE(), albu.ToGray()], p=1)], p=0.5),
            albu.GaussNoise(),
        ])

    train, val, test = getDataloaders(args.batch_size, args.root, frames_transforms, frames_aug_transforms,
                                      co_aug_transforms)
    train_length = len(train)
    epochs = args.steps // train_length

    tb_frames_train = next(iter(train))[0][0:1].to(device)
    tb_frames_val = next(iter(val))[0][0:1].to(device)
    tb_frames_test = next(iter(test))[0][0:1].to(device)

    os.makedirs(os.path.join("Checkpoints", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs", path), flush_secs=20)
    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", path, 'training_state.pt')) and False:
        checkpoint = torch.load(os.path.join("Checkpoints", path, 'training_state.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        #optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    
    save_checkpoint = True
    
    mile_stone = 10
    for e in range(starting_epoch, epochs):

        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        smooth_loss, bce_loss, total_loss = epoch(mymodel, train, loss_fnc, e,optim)

        if save_checkpoint:
            torch.save({
                'epoch': e,
                'model_state_dict': mymodel.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join("Checkpoints", path, 'training_state.pt'))

        smooth_loss_val, bce_loss_val, total_loss_val = epoch(mymodel, val, loss_fnc, -1)

        if total_loss_val < best_loss and save_checkpoint:
            print("---------saving new weights!----------") 
            best_loss = total_loss_val
            torch.save({
                'model_state_dict': mymodel.state_dict(),
                'loss_val': total_loss_val, 'smooth_loss_val': smooth_loss_val, 'bce_loss_val': bce_loss_val,
                'loss': total_loss, 'smooth_loss': smooth_loss, 'bce_loss': bce_loss,
            }, os.path.join("model_weight", path, 'best_weight.pt'))

        smooth_loss_test, bce_loss_test, total_loss_test = epoch(mymodel, test, loss_fnc,-1)
        # with torch.no_grad():
        #     mymodel.eval()
        #     pred_flow = mymodel.predictor(tb_frames_train)[0]
        #     tb.add_images('train', disp_function(pred_flow, tb_frames_train[0]), e, dataformats='NHWC')

        #     pred_flow = mymodel.predictor(tb_frames_val)[0]
        #     tb.add_images('val', disp_function(pred_flow, tb_frames_val[0]), e, dataformats='NHWC')

        #     pred_flow = mymodel.predictor(tb_frames_test)[0]
        #     tb.add_images('test', disp_function(pred_flow, tb_frames_test[0]), e, dataformats='NHWC')

        tb.add_scalars('loss', {"train": total_loss, "val": total_loss_val, "test": total_loss_test}, e)
        tb.add_scalars('smooth_loss', {"train": smooth_loss, "val": smooth_loss_val, "test": smooth_loss_test}, e)
        tb.add_scalars('bce_loss', {"train": bce_loss, "val": bce_loss_val, "test": bce_loss_test}, e)

        if "Flying" in args.root and e > 2:
            if e % mile_stone == 0:
                optim.param_groups[0]['lr'] *= 0.5
    tb.close()
