import torch
import torch.nn as nn


class EncoderProcessorDecoder(nn.Module):
    def __init__(
            self,
            encoder,
            processor,
            decoder,
            input_preprocessor=None,
            output_postprocessor=None,
    ):
        super(EncoderProcessorDecoder, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor

    def forward(self, inputs, decoder_query=None, T_out=None):
        trajectory = []
        # Operating input_preprocessor.
        if self.input_preprocessor:
            inputs_for_encoder = self.input_preprocessor(inputs, network_input_is_1d=True)
        else:
            inputs_for_encoder = inputs
        inputs_for_encoder = inputs_for_encoder.permute(2, 0, 1)
        inputs_for_encoder = inputs_for_encoder[:132]
        inputs_for_encoder = inputs_for_encoder.permute(1, 2, 0)
        # Operating encoder.
        z = self.encoder(inputs_for_encoder)

        if T_out is not None:
            for _ in range(T_out):
                # Operating processor at each time steps
                z = self.processor(z)

                # Operating decoder at each time steps
                output = self.decoder(decoder_query, z)
                trajectory.append(output)

            trajectory = torch.cat(trajectory, dim=-1)

            return trajectory

        else:
            # Operator processor.
            z = self.processor(z)

            # Operating decoder.
            if decoder_query is not None:
                output = self.decoder(decoder_query, z)
            else:
                output = self.decoder(z=z)

            return output