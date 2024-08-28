import torch
from torch import nn
from Baselines.ipot import *
from Baselines.ipot_epd import EncoderProcessorDecoder

class PhysicsEngine(nn.Module):
    def __init__(self,
                 device,
                 input_channel=3,
                 pos_channel=2,
                 num_bands=[64],
                 max_resolution=[64],
                 num_latents=64,
                 latent_channel = 16,
                 self_per_cross_attn=6,
                 cross_heads_num = 8,
                 self_heads_num = 8,
                 cross_heads_channel = None,
                 self_heads_channel = None,
                 ff_mult = 4,
                 latent_init_scale = 0.02,
                 output_scale = 0.1,
                 output_channel = 3,
                 position_encoding_type = 'pos2fourier'
                 ):
        super(PhysicsEngine, self).__init__()
        self.window_size = 5
        ipot_input_preprocessor = IPOTBasicPreprocessor(
                position_encoding_type=position_encoding_type,
                in_channel=input_channel,
                pos_channel=pos_channel,
                pos2fourier_position_encoding_kwargs=dict(
                    num_bands=num_bands,
                    max_resolution=max_resolution,
                )
            )
        # Encoder
        ipot_encoder = IPOTEncoder(
            input_channel=input_channel + (2 * sum(num_bands) + len(num_bands)),  # pos2fourier
            num_latents=num_latents,
            latent_channel=latent_channel,
            cross_heads_num=cross_heads_num,
            cross_heads_channel=cross_heads_channel,
            latent_init_scale=latent_init_scale
            )
        # Processor
        ipot_processor = IPOTProcessor(
            self_per_cross_attn=self_per_cross_attn,
            self_heads_channel=self_heads_channel,
            latent_channel=latent_channel,
            self_heads_num=self_heads_num,
            ff_mult=ff_mult,
            )
        # Decoder
        ipot_decoder = IPOTDecoder(
                output_channel=output_channel,
                query_channel=2 * sum(num_bands) + len(num_bands),  # pos2fourier
                latent_channel=latent_channel,
                cross_heads_num=cross_heads_num,
                cross_heads_channel=cross_heads_channel,
                ff_mult=ff_mult,
                output_scale=output_scale,
                position_encoding_type=position_encoding_type,
                pos2fourier_position_encoding_kwargs=dict(
                    num_bands=num_bands,
                    max_resolution=max_resolution, )
            )
        self.model = EncoderProcessorDecoder(
                encoder=ipot_encoder,
                processor=ipot_processor,
                decoder=ipot_decoder,
                input_preprocessor=ipot_input_preprocessor
        )
    def forward(self, data):
        
        in_data = data.pos.unsqueeze(0)
        output = self.model(in_data, decoder_query=data.recent_pos)
        output = output.squeeze(0)
        

        return output
    

