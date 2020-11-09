import torch
import torch.nn as nn


class Upsampler(nn.Module):

  def __init__(self, 
               upscale_per_level=2,
               layers=8,
               global_channels=16,
               channels_convolutions=[16, 16, 16, 16, 16, 16, 16, 16],
               kernel_size=2):
    super(Upsampler, self).__init__()

    self.upscale_per_level = upscale_per_level
    self.layers = layers
    self.global_channels = global_channels
    self.channels_convolutions = channels_convolutions
    self.kernel_size = kernel_size
    self.transposed_conv_tanh = nn.ModuleList([])
    self.transposed_conv_gate = nn.ModuleList([])
    self.reduce_conv1ds_res = nn.ModuleList([])
    self.reduce_conv1ds_skip = nn.ModuleList([])
    self.upsamplers_skip = nn.ModuleList([])

    self.upsampler_residual = nn.Upsample(scale_factor=self.upscale_per_level, 
                                          mode='nearest')

    self.total_upsampling = self.upscale_per_level ** self.layers

    current_upsampling = self.total_upsampling

    for i in range(self.layers):
      current_upsampling /= 2 
      if i == 0:
        in_channels = 1
      else:
        in_channels = self.global_channels

      t_conv_tanh = nn.ConvTranspose1d(in_channels=in_channels, 
                                     out_channels=self.channels_convolutions[i],
                                     stride=(self.upscale_per_level,),
                                     kernel_size=(self.kernel_size,),
                                     bias=True,
                                     padding=(
                                       (self.kernel_size - self.upscale_per_level)
                                        // 2,))

      t_conv_gate = nn.ConvTranspose1d(in_channels=in_channels,
                                     out_channels=self.channels_convolutions[i],
                                     stride=(self.upscale_per_level,),
                                     kernel_size=(self.kernel_size,),
                                     bias=True,
                                     padding=(
                                       (self.kernel_size - self.upscale_per_level) 
                                       // 2,))

      upsample_skip = nn.Upsample(scale_factor=current_upsampling, mode='nearest')
      reduce_conv_res = nn.Conv1d(in_channels=self.channels_convolutions[i],
                              out_channels=self.global_channels,
                              kernel_size=(1,),
                              bias=False)

      reduce_conv_skip = nn.Conv1d(in_channels=self.channels_convolutions[i],
                              out_channels=self.global_channels,
                              kernel_size=(1,),
                              bias=False)

      self.transposed_conv_tanh.append(t_conv_tanh)
      self.transposed_conv_gate.append(t_conv_gate)
      self.upsamplers_skip.append(upsample_skip)
      self.reduce_conv1ds_res.append(reduce_conv_res)
      self.reduce_conv1ds_skip.append(reduce_conv_skip)

    self.final_reduce = nn.Conv1d(in_channels=self.global_channels,
                                  out_channels=1,
                                  kernel_size=(1,),
                                  bias=True)

    self.act_tanh = nn.Tanh()
    self.act_sigmoid = nn.Sigmoid()
    self.act_relu = nn.ReLU()
  
  def forward(self, data_in):
    data_l = data_in

    skip_sum = torch.zeros((data_l.shape[0], 
                            self.global_channels, 
                            data_l.shape[2] * self.total_upsampling)).cuda()

    for i in range(self.layers):
      l_tanh = self.transposed_conv_tanh[i](data_l)
      l_gate = self.transposed_conv_gate[i](data_l)
      l_mult = self.act_tanh(l_tanh) * self.act_sigmoid(l_gate)

      reduced_skip = self.reduce_conv1ds_skip[i](l_mult)
      reduced_res = self.reduce_conv1ds_res[i](l_mult)
      upsampled_skip = self.upsamplers_skip[i](reduced_skip)
      skip_sum += upsampled_skip

      if i == 0:
        data_l = reduced_res
      else:
        upsampled_res = self.upsampler_residual(data_l)
        data_l = upsampled_res + reduced_res

    return self.final_reduce(skip_sum)


      
