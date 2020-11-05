import torch
import torch.nn as nn
import numpy as np


class ConditionalWaveNet(nn.Module):

  def __init__(self, num_dilated_layers=9, num_filters_glob=64,
               has_global_cond=False,
               glob_cond_size=100,
               has_local_cond=False,
               dialated_filter_nums=[96, 96, 96, 96, 96, 128, 128, 128, 128],
               out_scaling_filter_num=128,
               output_steps=256):
    super(ConditionalWaveNet, self).__init__()

    self.num_filters_glob = num_filters_glob
    self.has_global_cond = has_global_cond
    self.has_local_cond = has_local_cond
    self.dialated_filter_nums = dialated_filter_nums
    self.glob_cond_size = glob_cond_size
    self.out_scaling_filter_num = out_scaling_filter_num
    self.output_steps = output_steps

    self.num_dialated_layers = num_dilated_layers

    # initialize causal convolution
    self.causal_padding = nn.ConstantPad1d((1, 0), 0)
    self.causal_conv = nn.Conv1d(in_channels=1, 
                                 out_channels=self.num_filters_glob,
                                 kernel_size=(2,),
                                 stride=(1,),
                                 dilation=(1,))

    # initialize dialated convolutions for tanh activation function and gates
    self.dialated_convs_tanh = nn.ModuleList([])
    self.dialated_convs_gate = nn.ModuleList([])
    self.dialated_convs_padd = nn.ModuleList([])

    for i in range(self.num_dialated_layers):
      dial_conv_padd = nn.ConstantPad1d((2**(i+1), 0), 0)
      dial_conv_1d_tanh = nn.Conv1d(in_channels=self.num_filters_glob,
                                    out_channels=self.dialated_filter_nums[i],
                                    kernel_size=(2,),
                                    stride=(1,),
                                    dilation=(2**(i+1),))

      dial_conv_1d_gate = nn.Conv1d(in_channels=self.num_filters_glob,
                                    out_channels=self.dialated_filter_nums[i],
                                    kernel_size=(2,),
                                    stride=(1,),
                                    dilation=(2**(i+1),))
      
      self.dialated_convs_padd.append(dial_conv_padd)
      self.dialated_convs_tanh.append(dial_conv_1d_tanh)
      self.dialated_convs_gate.append(dial_conv_1d_gate)

    # initialize global conditioning if enabled
    if self.has_global_cond:
      self.glob_cond_syns_tanh = nn.ModuleList([])
      self.glob_cond_syns_gate = nn.ModuleList([])

      for i in range(self.num_dialated_layers):
        glob_cond_syn_tanh = nn.Linear(in_features=self.glob_cond_size, 
                                  out_features=self.dialated_filter_nums[i],
                                  bias=False)
        
        glob_cond_syn_gate = nn.Linear(in_features=self.glob_cond_size, 
                                  out_features=self.dialated_filter_nums[i],
                                  bias=False)

        self.glob_cond_syns_tanh.append(glob_cond_syn_tanh)
        self.glob_cond_syns_gate.append(glob_cond_syn_gate)

    # initialize local conditioning if enabled
    if self.has_local_cond:
      self.local_cond_syns_tanh = nn.ModuleList([])
      self.local_cond_syns_gate = nn.ModuleList([])
      self.local_cond_padds = nn.ModuleList([])

      for i in range(self.num_dialated_layers):
        local_cond_padd = nn.ConstantPad1d((2**(i+1), 0), 0)
        local_cond_syn_tanh = nn.Conv1d(in_channels=1,
                                        out_channels=self.dialated_filter_nums[i],
                                        kernel_size=(2,),
                                        stride=(1,),
                                        dilation=(2**(i+1),))

        local_cond_syn_gate = nn.Conv1d(in_channels=1,
                                        out_channels=self.dialated_filter_nums[i],
                                        kernel_size=(2,),
                                        stride=(1,),
                                        dilation=(2**(i+1),))

        self.local_cond_padds.append(local_cond_padd)
        self.local_cond_syns_tanh.append(local_cond_syn_tanh)
        self.local_cond_syns_gate.append(local_cond_syn_gate)

    # initialize scaling 1x1 convolution Kernels
    self.scaling_convs = nn.ModuleList([])
    for i in range(self.num_dialated_layers):
      scale_conv_1d = nn.Conv1d(in_channels=self.dialated_filter_nums[i],
                                out_channels=self.num_filters_glob,
                                kernel_size=(1,),
                                stride=(1,))

      self.scaling_convs.append(scale_conv_1d)
    
    self.pre_output_scaler = nn.Conv1d(in_channels=self.num_filters_glob,
                                       out_channels=self.out_scaling_filter_num,
                                       kernel_size=(1,),
                                       stride=(1,))

    self.output_scaler = nn.Conv1d(in_channels=self.out_scaling_filter_num,
                                   out_channels=self.output_steps,
                                   kernel_size=(1,),
                                   stride=(1,))

    self.act_tanh = nn.Tanh()
    self.act_relu = nn.ReLU()
    self.act_sigmoid = nn.Sigmoid()
    self.act_softmax = nn.Softmax(dim=1)

  def causal(self, data):
    padded_d = self.causal_padding(data)
    return self.causal_conv(padded_d)

  def dilated(self, data, h=None, y=None):
    l_data = data
    l_skip_sum = torch.zeros(l_data.shape).cuda()

    for i in range(self.num_dialated_layers):
      l_data, l_skip = self.single_cell(l_data, i, h=h, y=y)
      l_skip_sum += l_skip
    
    return l_data, l_skip_sum

  def foot(self, data, h=None, y=None):
    c_out = self.causal(data)
    d_out, d_skip = self.dilated(c_out, h=h, y=y)

    return d_out, d_skip

  def head(self, skip_sum):
    r_skip_s = self.act_relu(skip_sum)
    p_o_scaled = self.pre_output_scaler(r_skip_s)
    r_skip_s2 = self.act_relu(p_o_scaled)
    o_scaled = self.output_scaler(r_skip_s2)
    return self.act_softmax(o_scaled)

  def forward(self, data, y=None, h=None):
    data_out, skip = self.foot(data, h=h, y=y)
    return self.head(skip)

  def single_cell(self, data_in, layer_idx, h=None, y=None):
    padded_d = self.dialated_convs_padd[layer_idx](data_in)
    l_dc_tanh = self.dialated_convs_tanh[layer_idx](padded_d)

    l_dc_gate = self.dialated_convs_gate[layer_idx](padded_d)

    if self.has_global_cond:
      g_cond_tanh = self.glob_cond_syns_tanh[layer_idx](h)
      g_cond_gate = self.glob_cond_syns_gate[layer_idx](h)
      l_dc_tanh += torch.reshape(g_cond_tanh, 
                                 (g_cond_tanh.shape[0], g_cond_tanh.shape[1],1))
      l_dc_gate += torch.reshape(g_cond_gate, 
                                 (g_cond_gate.shape[0], g_cond_gate.shape[1],1))

    if self.has_local_cond:
      padded_y = self.local_cond_padds[layer_idx](y)
      l_cond_tanh = self.local_cond_syns_tanh[layer_idx](padded_y)
      l_cond_gate = self.local_cond_syns_gate[layer_idx](padded_y)

      l_dc_tanh += l_cond_tanh
      l_dc_gate += l_cond_gate

    l_mult = self.act_tanh(l_dc_tanh) * self.act_sigmoid(l_dc_gate)
    l_mult_scaled = self.scaling_convs[layer_idx](l_mult)
    print(data_in.shape, l_mult_scaled.shape, l_mult.shape)
    l_residual = data_in + l_mult_scaled

    return l_residual, l_mult_scaled
