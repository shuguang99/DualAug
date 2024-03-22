import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_bn_momentum = 0.1
CpG = 8


class ExampleWiseBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
            local_means = input.mean([2, 3])
            local_global_means = local_means + (mean.unsqueeze(0) - local_means).detach()
            local_vars = input.var([2, 3], unbiased=False)
            local_global_vars = local_vars + (var.unsqueeze(0) - local_vars).detach()
            input = (input - local_global_means[:,:,None,None]) / (torch.sqrt(local_global_vars[:,:,None,None] + self.eps))
        else:
            mean = self.running_mean
            var = self.running_var
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class VirtualBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
            input = (input - mean.detach()[None, :, None, None]) / (torch.sqrt(var.detach()[None, :, None, None] + self.eps))
        else:
            mean = self.running_mean
            var = self.running_var
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, norm_creator, stride=1, adaptive_dropouter_creator=None):
        super(WideBasic, self).__init__()
        self.bn1 = norm_creator(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        if adaptive_dropouter_creator is None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = adaptive_dropouter_creator(planes, 3, stride, 1)
        self.bn2 = norm_creator(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, adaptive_dropouter_creator, adaptive_conv_dropouter_creator, groupnorm, examplewise_bn, virtual_bn):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.adaptive_conv_dropouter_creator = adaptive_conv_dropouter_creator

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        assert sum([groupnorm,examplewise_bn,virtual_bn]) <= 1
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.adaptive_dropouters = [] #nn.ModuleList()

        if groupnorm:
            print('Uses group norm.')
            self.norm_creator = lambda c: nn.GroupNorm(max(c//CpG, 1), c)
        elif examplewise_bn:
            print("Uses Example Wise BN")
            self.norm_creator = lambda c: ExampleWiseBatchNorm2d(c, momentum=_bn_momentum)
        elif virtual_bn:
            print("Uses Virtual BN")
            self.norm_creator = lambda c: VirtualBatchNorm2d(c, momentum=_bn_momentum)
        else:
            self.norm_creator = lambda c: nn.BatchNorm2d(c, momentum=_bn_momentum)

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = self.norm_creator(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        if adaptive_dropouter_creator is not None:
            last_dropout = adaptive_dropouter_creator(nStages[3])
        else:
            last_dropout = lambda x: x
        self.adaptive_dropouters.append(last_dropout)
        self.nChannels = nStages[3]

        # self.sigma_lambda = nn.Parameter(torch.ones([]))
        # self.apply(conv_init)

    def to(self, *args, **kwargs):
        super().to(*args,**kwargs)
        print(*args)
        for ad in self.adaptive_dropouters:
            if hasattr(ad,'to'):
                ad.to(*args,**kwargs)
        return self

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for i,stride in enumerate(strides):
            ada_conv_drop_c = self.adaptive_conv_dropouter_creator if i == 0 else None
            new_block = block(self.in_planes, planes, dropout_rate, self.norm_creator, stride, adaptive_dropouter_creator=ada_conv_drop_c)
            layers.append(new_block)
            if ada_conv_drop_c is not None:
                self.adaptive_dropouters.append(new_block.dropout)

            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.adaptive_dropouters[-1](out)
        out = self.linear(out)

        return out

    def intermediate_forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        return out

    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list
    def dice(self, x, p, info):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.adaptive_dropouters[-1](out)
        contrib = info[None, :] * self.linear.weight.data.cpu().numpy()
        thresh = np.percentile(contrib, p)
        mask = torch.Tensor((contrib > thresh))
        masked_w = (self.linear.weight.squeeze().cpu() * mask).cuda()
        vote = out[:, None, :] * masked_w.cuda()
        if self.linear.bias is not None:
            out = vote.sum(2) + self.linear.bias
        else:
            out = vote.sum(2)
        return out
    def features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.adaptive_dropouters[-1](out)

        return out

class WideBasic2(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
class WideResNet2(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet2, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic2, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic2, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic2, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
class WideResNetCosine(WideResNet2):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNetCosine, self).__init__(
            depth, widen_factor, dropout_rate, num_classes
        )
        self.fc = nn.Linear(64*widen_factor, num_classes, bias=False)
        self.fc_w = nn.Parameter(self.fc.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(64*widen_factor, 1)

    def forward(
        self, x, y=None, mixup=None, alpha=None, all_pred=False,
        candidate_layers=[0, 1, 2, 3],
    ):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        scale = torch.exp(self.bn_scale(self.fc_scale(out)))
        x_norm = F.normalize(out)
        w_norm = F.normalize(self.fc_w)
        w_norm_transposed = torch.transpose(w_norm, 0, 1)

        cos_sim = torch.mm(x_norm, w_norm_transposed)  # cos_theta
        scaled_cosine = cos_sim * scale
        softmax = F.softmax(scaled_cosine, 1)

        return scaled_cosine, softmax, scale, cos_sim