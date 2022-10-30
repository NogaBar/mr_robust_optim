import argparse
import dataset
import torch
import utils
from resnet import *
from wide_resnet import *
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import os
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import json
import tqdm

from dataset_clothing1m import Clothing1M

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=160, type=int, help='number of epochs')
parser.add_argument('--milestones', default=[80, 120], type=int, nargs='+',help='epochs forwhich the leaning rate changes')
parser.add_argument('--gamma', default=0.1, type=float, help='multiplicative factor of learning rate')
parser.add_argument('--weight-decay', default=0.0001, type=float, help='multiplicative factor of learning rate')
parser.add_argument('--num-workers', default=1, type=int, help='num workers')
parser.add_argument('--cuda-device', default=0, type=str, help='gpu device number')
parser.add_argument('--save-freq', default=-1, type=int, help='save model frequency, -1 for saving only the last model')

parser.add_argument('--noise-ratio', default=0.4, type=float, help='fraction of noisy train labels')
parser.add_argument('--weight-update', default='mr', type=str,
                    help='mr or none')
parser.add_argument('--mr-lr', default=0.01, type=float, help='eta is the step size of the MW algorithms')
parser.add_argument('--gamma-mr', default=1., type=float, help='scale eta')
parser.add_argument('--mr-milestones', default=[30, 80, 120],  type=int, nargs='+', help='specify when to scale eta')
parser.add_argument('--mixup', default=0,  type=float, help='parameter for generating mixup examples')
parser.add_argument('--label-smoothing',action='store_true', help='smooth label')
parser.add_argument('--mid-update',action='store_true', help='upadate weighting in the middle of epochs')
parser.add_argument('--max_weight', default=1., type=float, help='maximal weight of each example')


parser.add_argument('--print-freq', default=10, type=int, help='print frequency') # TODO: fix this

parser.add_argument('--dataset', default='cifar10', help='specify dataset - cifar10 or cifar100')
parser.add_argument('--model', default='resnet18', help='model to run')
parser.add_argument('--optim', default='sgd', help='optimizer to use [sgd or adam]')

parser.add_argument('--checkpoint-dir', default='', type=str, help='name of directory')

parser.add_argument('--resume', default='', type=str, help='resume from checkpoint path')

args = parser.parse_args()
os.makedirs(args.checkpoint_dir, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ", device)


# Data
if 'cifar' in args.dataset:
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    train_data, test_data, train_data_fw = \
        dataset.indexed_cifar(args.batch_size, args.noise_ratio, dataset_name=args.dataset, num_cls=num_classes)
else:
    train_data = DataLoader(Clothing1M('train'), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_data_fw = DataLoader(Clothing1M('train'), batch_size=256, num_workers=args.num_workers, shuffle=True)
    test_data = DataLoader(Clothing1M('test'), batch_size=256, num_workers=args.num_workers, shuffle=False)
    num_classes = 14

# Loss
def weighted_loss(y, y_hat, weighting):
    part = torch.nn.CrossEntropyLoss(reduction='none')
    loss = part(y, y_hat)
    return loss * weighting

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, weighting):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.sum(-true_dist * pred, dim=self.dim) * weighting

criterion = weighted_loss
if args.label_smoothing:
    criterion = LabelSmoothingLoss(num_classes, 0.1)
# Model
if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    if args.model == 'resnet18':
        model = ResNet18(num_classes)
    elif args.model == 'resnet34':
        model = ResNet34(num_classes)
    elif args.model == 'wrn28_10':
        model = WRN28_10(num_classes)
    else:
        raise NotImplementedError
else:
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True, progress=False)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, progress=False)
    else:
        raise NotImplementedError

    print('Using pre-trained model, change number of output classes to match clothing1m')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

if args.resume is not '':
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
    print('==> resume from file...')
    checkpoint = torch.load(args.resume)

    model.load_state_dict(checkpoint['model'])
    model.to(device)


# Optimizer
# eta = args.mr_lr
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)


def save_model(epoch, test_acc, test_loss):
    print('\t==> saving model ...')
    checkpoint = {'epoch': epoch,
                  'test_acc': test_acc,
                  'test_loss': test_loss,
                  'model': model.state_dict()}
    path = os.path.join(args.checkpoint_dir, 'ckpt{}.pth'.format(epoch+1))
    torch.save(checkpoint, path)


def update_weighting(loss, eta):
  weights = torch.exp(-eta * loss) / torch.exp(-eta * loss).sum()
  return weights


def evaluate():
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for input_, target in test_data:
            input_, target = input_.to(device), target.to(device)

            outputs = model(input_)
            loss += torch.nn.CrossEntropyLoss()(outputs, target).item() * target.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    model.eval_mode = False
    return correct / total, loss / total

def loss_per_input(shape):
    curr_loss = torch.zeros(shape, device=device)
    with torch.no_grad():
        for inputs, labels, index in train_data_fw:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            curr_loss[index] =\
                torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)
    return curr_loss


def update_total_loss_weighting(curr_loss, total_loss, eta):
    if args.weight_update == 'mr':
        total_loss += curr_loss
        weighting = update_weighting(total_loss, eta)
    elif args.weight_update == 'none':
        weighting = torch.ones_like(curr_loss) / len(curr_loss)
    else:
        raise NotImplementedError
    return total_loss, weighting



def train_mr(eta):
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    n_samples = len(train_data.dataset)
    is_clothing = isinstance(train_data.dataset, Clothing1M)

    weighting = torch.ones((n_samples,), dtype=torch.float32, device=device) * 1 / n_samples
    total_loss = torch.zeros_like(weighting)

    noisy_weight = []
    noisy_loss = []
    clean_loss = []
    for epoch in tqdm.tqdm(range(0,args.epochs)):  # loop over the dataset multiple times
        model.train()
        for i, data in enumerate(train_data, 0):
            inputs, labels, index = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels, weighting[index])

            # normilize by the sum of all weights in the batch
            loss = loss.sum() * 1 / weighting[index].sum()
            loss.backward()
            optimizer.step()

            if args.mid_update and i == len(train_data) // 2 and epoch > 0:
                curr_loss = loss_per_input(model, train_data_fw, device, weighting.shape)
                total_loss, weighting = update_total_loss_weighting(curr_loss, total_loss, eta)


        scheduler.step()
        noisy_i = [] if is_clothing else train_data.dataset.noisy_i
        noisy_n = 0 if is_clothing else train_data.dataset.noisy_n
        noisy_weight.append(weighting[noisy_i].sum().item())

        curr_loss = loss_per_input(weighting.shape)

        # update eta
        if epoch + 1 in args.mr_milestones:
            eta *= args.gamma_mr

        total_loss, weighting = update_total_loss_weighting(curr_loss, total_loss, eta)
        if args.max_weight < 1.:
            actual_max_weight = 1/(args.max_weight*n_samples)
            while weighting.max() > actual_max_weight:
                res = torch.clamp(weighting - actual_max_weight, min=0.).sum()
                normalized_low = weighting[weighting < actual_max_weight] / weighting[weighting < actual_max_weight].sum()
                weighting[weighting < actual_max_weight] = weighting[weighting < actual_max_weight] \
                                                               + normalized_low * res
                weighting[weighting > actual_max_weight] = actual_max_weight
                print('in loop normalizing weights')
                print(f'max weight * N: {weighting.max()*n_samples}, min: {weighting.min()*n_samples}')

        noisy_loss.append(curr_loss[noisy_i].mean().item())
        curr_clean_loss = (curr_loss.sum() - curr_loss[noisy_i].sum()) / \
                          (n_samples - noisy_n)
        clean_loss.append(curr_clean_loss.item())

        test_acc, test_loss = evaluate()
        train_loss_list.append((curr_loss * weighting).sum().item())
        if epoch % args.print_freq == args.print_freq - 1:
            print('Epoch #{} (TRAIN): loss={:.2f}\t(TEST) loss={:.2f}\tacc={:.2f}'
                  .format(epoch + 1, (curr_loss * weighting).sum().item(), test_loss, test_acc*100))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if (args.save_freq > 0 and epoch % args.save_freq == args.save_freq - 1) or epoch == args.epochs - 1:
            save_model(epoch, test_acc_list, test_loss_list)
    return train_loss_list, test_loss_list, test_acc_list, noisy_loss, clean_loss, noisy_weight


def mixup_data(x, y):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    alpha = args.mixup
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def train_mr_mixup(eta):
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    n_samples = len(train_data.dataset)
    is_clothing = isinstance(train_data.dataset, Clothing1M)

    weighting = torch.ones((n_samples,), dtype=torch.float32, device=device) * 1 / n_samples
    total_loss = torch.zeros_like(weighting)

    noisy_weight = []
    noisy_loss = []
    clean_loss = []
    for epoch in tqdm.tqdm(range(args.epochs)):  # loop over the dataset multiple times
        model.train()
        for i, data in enumerate(train_data, 0):
            inputs, labels, index = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs, targets_a, targets_b, lam, perm_index = mixup_data(inputs, labels)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a, weighting[index]) + \
                   (1 - lam) * criterion(outputs, targets_b, weighting[index[perm_index]])

            # normilize by the sum of all weights in the batch
            loss = loss.sum() * 1 / weighting[index].sum()
            loss.backward()
            optimizer.step()

        scheduler.step()
        noisy_i = [] if is_clothing else train_data_fw.dataset.noisy_i
        noisy_n = 0 if is_clothing else train_data_fw.dataset.noisy_n

        noisy_weight.append(weighting[noisy_i].sum().item())

        curr_loss = loss_per_input(weighting.shape)
        # update eta
        if epoch + 1 in args.mr_milestones:
            eta *= args.gamma_mr

        total_loss, weighting = update_total_loss_weighting(curr_loss, total_loss, eta)
        if args.max_weight < 1.:
            actual_max_weight = 1/(args.max_weight*n_samples)
            while weighting.max() > actual_max_weight:
                res = torch.clamp(weighting - actual_max_weight, min=0.).sum()
                normalized_low = weighting[weighting < actual_max_weight] / weighting[weighting < actual_max_weight].sum()
                weighting[weighting < actual_max_weight] = weighting[weighting < actual_max_weight] \
                                                               + normalized_low * res
                weighting[weighting > actual_max_weight] = actual_max_weight
                print('in loop normalizing weights')
                print(f'max weight * N: {weighting.max()*n_samples}, min: {weighting.min()*n_samples}')


        noisy_loss.append(curr_loss[noisy_i].mean().item())
        curr_clean_loss = (curr_loss.sum() - curr_loss[noisy_i].sum()) / \
                          (n_samples - noisy_n)
        clean_loss.append(curr_clean_loss.item())

        test_acc, test_loss = evaluate()
        train_loss_list.append((curr_loss * weighting).sum().item())
        if epoch % args.print_freq == args.print_freq - 1:
            print('Epoch #{} (TRAIN): loss={:.2f}\t(TEST) loss={:.2f}\tacc={:.2f}'
                  .format(epoch + 1, (curr_loss * weighting).sum().item(), test_loss, test_acc*100))

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if (args.save_freq > 0 and epoch % args.save_freq == args.save_freq - 1) or epoch == args.epochs - 1:
            save_model(epoch, test_acc_list, test_loss_list)
    return train_loss_list, test_loss_list, test_acc_list, noisy_loss, clean_loss, noisy_weight

if __name__ == '__main__':
    # dumping args to txt file
    with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    eta = args.mr_lr
    if args.mixup == 0:
        train_loss, test_loss, test_acc, noisy_loss, clean_loss, noisy_weight = train_mr(eta)
    else:
        train_loss, test_loss, test_acc, noisy_loss, clean_loss, noisy_weight = train_mr_mixup(eta)
