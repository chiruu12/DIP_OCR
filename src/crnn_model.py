import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, rnn_layers=2, cnn_output_height=32):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )

        self.map_to_sequence = nn.Linear(512 * 2, rnn_hidden_size)

        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Linear(rnn_hidden_size * 2, num_chars + 1)

    def forward(self, x):
        conv_features = self.cnn(x)

        batch, channels, height, width = conv_features.size()
        sequence = conv_features.permute(0, 3, 1, 2).contiguous()
        sequence = sequence.view(batch, width, channels * height)

        sequence = self.map_to_sequence(sequence)

        rnn_output, _ = self.rnn(sequence)

        output = self.classifier(rnn_output)

        # --- THIS IS THE FIX ---
        # The model should return the (Batch, SequenceLength, NumClasses) tensor.
        # The training loop will handle the permutation for the loss function.
        return output