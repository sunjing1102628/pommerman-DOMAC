import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_generic import NNBase
import numpy as np


class ConvNet3(nn.Module):
    def __init__(
            self, input_shape, num_channels=64, output_size=512,
            batch_norm=True, activation_fn=F.relu, dilation=True):
        super(ConvNet3, self).__init__()

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.activation_fn = activation_fn
        self.flattened_size = num_channels * (input_shape[1] - 2) * (input_shape[2] - 2)
        self.drop_prob = 0.2

        self.conv1 = nn.Conv2d(input_shape[0], num_channels, 5, stride=1, padding=2)
        if dilation:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, dilation=2, padding=2)
        else:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation_fn(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.activation_fn(x)
        out = self.fc2(x)

        return out


class ConvNet4(nn.Module):
    #print('cov4!!!')
    def __init__(self, input_shape, num_channels=64, output_size=512,
                 batch_norm=True, activation_fn=F.relu, dilation=False):
        super(ConvNet4, self).__init__()

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.activation_fn = activation_fn
        self.flattened_size = num_channels * (input_shape[1] - 4) * (input_shape[2] - 4)
        self.drop_prob = 0.2

        self.conv1 = nn.Conv2d(input_shape[0], num_channels, 3, stride=1, padding=1)
        if dilation:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, dilation=2, padding=2)
        else:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            self.bn3 = nn.BatchNorm2d(num_channels)
            self.bn4 = nn.BatchNorm2d(num_channels)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x
            self.bn4 = lambda x: x

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation_fn(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation_fn(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.activation_fn(x)
        #x = F.dropout(x, p=self.drop_prob, training=self.training)
        out = self.fc2(x)

        return out


class PommNet(NNBase):

    def __init__(self, obs_shape,recurrent=False, hidden_size=512, batch_norm=True, cnn_config='conv4'):
        super(PommNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape #(1092,)
        # self.action_shape= action_shape
        # print('self.action_shape',action_shape)
        #print('PommNet!!!!')


        # FIXME hacky, recover input shape from flattened observation space
        # assuming an 11x11 board and 3 non spatial features
        bs = 11
        self.other_shape = [3]
        #print('obs_shape[0]',obs_shape[0]) #1092
        input_channels = (obs_shape[0] - self.other_shape[0]) // (bs*bs) #9
        self.image_shape = [input_channels, bs, bs] #[9, 11, 11]

        assert np.prod(obs_shape) >= np.prod(self.image_shape)

        if cnn_config == 'conv3':

            self.common_conv = ConvNet3(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)
        else:
            #print('conv4')
            assert cnn_config == 'conv4'
            self.common_conv = ConvNet4(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)

        self.common_mlp = nn.Sequential(
            nn.Linear(self.other_shape[0], hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU()
        )


        self.actor = nn.Linear(hidden_size + hidden_size//4, hidden_size)
        '''self.actor = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//4, hidden_size),
            nn.Linear(hidden_size, action_shape)
        )'''
        self.critic = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//4, 1),
            nn.Tanh()
        )

    def forward(self, inputs,ids, rnn_hxs, masks):

        inputs_image = inputs[:, :-self.other_shape[0]].view([-1] + self.image_shape)
        inputs_other = inputs[:, -self.other_shape[0]:]

        x_conv = self.common_conv(inputs_image)
        x_mlp = self.common_mlp(inputs_other)
        x = torch.cat([x_conv, x_mlp], dim=1)
        #print('x_len',len(x))
        #x = x_conv + x_mlp

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        out_actor = self.actor(x)
        #out_actor = F.softmax(self.actor(x), dim=-1)
        #print('out_actor',out_actor)
        out_value = self.critic(x)


        return out_value, out_actor, rnn_hxs

class ActorNet(NNBase):

    def __init__(self, obs_shape,recurrent=False, hidden_size=512, batch_norm=True, cnn_config='conv4'):
        super(ActorNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape #(1092,)
        # self.action_shape= action_shape
        # print('self.action_shape',action_shape)
        #print('PommNet!!!!')


        # FIXME hacky, recover input shape from flattened observation space
        # assuming an 11x11 board and 3 non spatial features
        bs = 11
        self.other_shape = [3]
        #print('obs_shape[0]',obs_shape[0]) #1092
        input_channels = (obs_shape[0] - self.other_shape[0]) // (bs*bs) #9
        self.image_shape = [input_channels, bs, bs] #[9, 11, 11]

        assert np.prod(obs_shape) >= np.prod(self.image_shape)

        if cnn_config == 'conv3':

            self.common_conv = ConvNet3(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)
        else:
            #print('conv4')
            assert cnn_config == 'conv4'
            self.common_conv = ConvNet4(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)

        self.common_mlp = nn.Sequential(
            nn.Linear(self.other_shape[0], hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU()
        )


        self.actor = nn.Linear(hidden_size + hidden_size//4, hidden_size)

        '''self.critic = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//4, 1),
            nn.Tanh()
        )'''

    def forward(self, inputs, rnn_hxs, masks):

        inputs_image = inputs[:, :-self.other_shape[0]].view([-1] + self.image_shape)
        inputs_other = inputs[:, -self.other_shape[0]:]

        x_conv = self.common_conv(inputs_image)
        x_mlp = self.common_mlp(inputs_other)
        x = torch.cat([x_conv, x_mlp], dim=1)
        #print('x_len',len(x))
        #x = x_conv + x_mlp

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        out_actor = self.actor(x)
        #out_actor = F.softmax(self.actor(x), dim=-1)
        #print('out_actor',out_actor)
        #out_value = self.critic(x)


        return out_actor, rnn_hxs


class CriticNet(NNBase):

    def __init__(self, obs_shape,action_num,agent_num,num_quant, recurrent=False, hidden_size=512, batch_norm=True, cnn_config='conv4'):
        super(CriticNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape  # (1092,)
        self.num_quant = num_quant
        self.num_actions= action_num
        # print('self.action_shape',action_shape)
        # print('PommNet!!!!')

        # FIXME hacky, recover input shape from flattened observation space
        # assuming an 11x11 board and 3 non spatial features
        bs = 11
        self.other_shape = [3]
        # print('obs_shape[0]',obs_shape[0]) #1092
        input_channels = (obs_shape[0] - self.other_shape[0]) // (bs * bs)  # 9
        # print('input_channels',input_channels)
        self.image_shape1 = [input_channels, bs, bs]
        self.image_shape = [agent_num*input_channels, bs, bs]  # [9, 11, 11]
        # print('self.image_shape',self.image_shape)
        # print('np.prod(obs_shape)',np.prod(obs_shape))
        # print('np.prod(self.image_shape)',np.prod(self.image_shape))

        assert np.prod(agent_num*obs_shape) >= np.prod(self.image_shape)

        if cnn_config == 'conv3':

            self.common_conv = ConvNet3(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)
        else:
            # print('conv4')
            assert cnn_config == 'conv4'
            self.common_conv = ConvNet4(
                input_shape=self.image_shape,
                output_size=hidden_size,
                batch_norm=batch_norm)

        self.common_mlp = nn.Sequential(
            nn.Linear(agent_num*self.other_shape[0], hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU()
        )



        self.critic = nn.Sequential(
            nn.Linear(1+agent_num+hidden_size + hidden_size//4, action_num*num_quant),
            nn.Tanh()
        )

    def forward(self, inputs, ids, rnn_hxs, masks,actions):
        # print('inputs.size',inputs.size())
        # print('inputs[0]',inputs[0].size())

        inputs_image0 = inputs[0][:, :-self.other_shape[0]].view([-1] + self.image_shape1)
        #print('inputs_image',inputs_image0.size())
        inputs_image2 = inputs[1][:, :-self.other_shape[0]].view([-1] + self.image_shape1)
        #print('inputs_image2',inputs_image2.size())
        inputs_image = torch.cat([inputs_image0,inputs_image2],dim= 1)
        #print('inputs_image',inputs_image.size())
        inputs_other0 = inputs[0][:, -self.other_shape[0]:]
        #print('inputs_other',inputs_other0.size())
        inputs_other1 = inputs[1][:, -self.other_shape[0]:]
       # print('inputs_other1', inputs_other1.size())
        inputs_other = torch.cat([inputs_other0,inputs_other1],dim= -1)
        #print('inputs_other',inputs_other.size())

        x_conv = self.common_conv(inputs_image)
        x_mlp = self.common_mlp(inputs_other)
        x1 = torch.cat([ids, actions], dim=-1)
        x = torch.cat([x1,x_conv, x_mlp], dim=-1)


        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        out_value = self.critic(x)
        #print('out_value',out_value.size())
        return out_value.view(-1, self.num_actions, self.num_quant)
