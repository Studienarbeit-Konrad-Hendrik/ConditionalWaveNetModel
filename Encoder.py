import torch
import torch.nn as nn
import numpy as np


class Downsampler(nn.Module):

  def __init__(self, num_conv_layers=8, 
               num_conv_channels=[16, 16, 16, 16, 16, 16, 16, 16], 
               reduction_per_layer=2,
               global_channels=16,
               kernel_size=4):
    super(Downsampler, self).__init__()
    
    self.num_conv_layers = num_conv_layers

    self.convolutions_tanh = nn.ModuleList([])
    self.convolutions_gate = nn.ModuleList([])
    self.skiplane_pools = nn.ModuleList([])
    self.reduce_1d_convs_res = nn.ModuleList([])
    self.reduce_1d_convs_skip = nn.ModuleList([])
    self.residual_pooling = nn.AvgPool1d(kernel_size=(reduction_per_layer,))
    self.act_relu = nn.ReLU()
    self.act_tanh = nn.Tanh()
    self.act_sigmoid = nn.Sigmoid()
    self.global_channels = global_channels

    self.overall_reduction = reduction_per_layer ** num_conv_layers
  	
    layer_reduction = self.overall_reduction

    for i in range(num_conv_layers):

      layer_reduction = layer_reduction // 2

      if i == 0:
        in_channels = 1
      else:
        in_channels = self.global_channels

      conv_tanh = nn.Conv1d(in_channels=in_channels, 
                       out_channels=num_conv_channels[i],
                       kernel_size=(kernel_size,),
                       stride=(reduction_per_layer,),
                       padding=((kernel_size-reduction_per_layer)//2,),
                       bias=True)

      conv_gate = nn.Conv1d(in_channels=in_channels, 
                       out_channels=num_conv_channels[i],
                       kernel_size=(kernel_size,),
                       stride=(reduction_per_layer,),
                       padding=((kernel_size-reduction_per_layer)//2,),
                       bias=True)

      conv_red_res = nn.Conv1d(in_channels=num_conv_channels[i],
                            out_channels=self.global_channels,
                            kernel_size=(1,),
                            bias=False)

      conv_red_skip = nn.Conv1d(in_channels=num_conv_channels[i],
                            out_channels=self.global_channels,
                            kernel_size=(1,),
                            bias=False)
      
      skiplane_pool = nn.AvgPool1d(kernel_size=(layer_reduction,))

      self.convolutions_tanh.append(conv_tanh)
      self.convolutions_gate.append(conv_gate)
      self.skiplane_pools.append(skiplane_pool)
      self.reduce_1d_convs_res.append(conv_red_res)
      self.reduce_1d_convs_skip.append(conv_red_skip)

    

    self.output_conv = nn.Conv1d(in_channels=self.global_channels, 
                                 out_channels=1,
                                 kernel_size=(1,),
                                 bias=True)

  def forward(self, data_in):
    data_l = data_in

    skip_sum = torch.zeros((data_l.shape[0], self.global_channels, 
                            data_l.shape[2] // self.overall_reduction)).cuda()

    for i in range(self.num_conv_layers):
      data_c_tanh = self.convolutions_tanh[i](data_l)
      data_c_gate = self.convolutions_gate[i](data_l)
      data_c = self.act_tanh(data_c_tanh) * self.act_sigmoid(data_c_gate)

      reduced_skip = self.reduce_1d_convs_skip[i](data_c)
      reduced_res = self.reduce_1d_convs_res[i](data_c)

      skip_value = self.skiplane_pools[i](reduced_skip)
      skip_sum += skip_value
      if i == 0:
        data_l = reduced_res
      else:
        data_l = self.residual_pooling(data_l) + reduced_res

    skip_single_channel = self.output_conv(skip_sum)

    # activated = self.act_relu(skip_single_channel)

    return skip_single_channel

    

