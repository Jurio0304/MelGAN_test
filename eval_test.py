import os
import sys

from mel2wav.dataset import AudioDataset
from mel2wav.modules import Generator, Discriminator, Audio2Mel
from mel2wav.utils import save_sample

import torch
from torch.utils.data import DataLoader

import yaml
import time
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default='eval_test_result')
    parser.add_argument("--load_path", default='test_result')
    parser.add_argument("--wavs_path", default='wavs')
    parser.add_argument("--n_mel_channels", type=int, default=40)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--n_test_samples", type=int, default=100)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    wavs_root = Path(args.wavs_path) if args.wavs_path else None
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)

    #######################
    # Load netG Models #
    #######################
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda()

    print(netG)


    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))

    #######################
    # Create data loaders #
    #######################
    test_voc = []
    test_audio = []

    if os.listdir(wavs_root):
        test_set = AudioDataset(
            "eval_test_file.txt",
            22050 * 4,
            sampling_rate=22050,
            augment=False,
        )

        test_loader = DataLoader(test_set, batch_size=1)

        ##########################
        # Dumping original audio #
        ##########################
        for i, x_t in enumerate(test_loader):
            x_t = x_t.cuda()
            s_t = fft(x_t).detach()

            test_voc.append(s_t.cuda())
            test_audio.append(x_t)

            audio = x_t.squeeze().cpu()
            save_sample(root / ("original_%d.wav" % i), 22050, audio)
            print("original_%d.wav stored" % i)

            if i == args.n_test_samples - 1:
                break

        # enable cudnn autotuner to speed up training
        torch.backends.cudnn.benchmark = True

        ###########################
        # Dumping genetated audio #
        ###########################
        st = time.time()
        with torch.no_grad():
            for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                pred_audio = netG(voc)
                pred_audio = pred_audio.squeeze().cpu()
                save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
                print("generated_%d.wav stored" % i)

        print("Took %5.4fs to generate samples" % (time.time() - st))
        print("-" * 10, "Decoding completed!", "-" * 10)

    else:
        print("Error! There are no reco_wavs!")

if __name__ == "__main__":
    main()

    sys.exit(0)


