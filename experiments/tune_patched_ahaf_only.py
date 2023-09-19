#!/usr/bin/env python3

from train_common import train_variant


def main():
    af_names = ("ReLU", "SiLU")
    start_ep = 100
    end_ep = start_ep + 50

    for af in af_names:
        train_variant(
            "KerasNet", "ahaf", "CIFAR-10", af_name=af,
            start_epoch=start_ep, end_epoch=end_ep, patched=True,
            tune_aaf=True
        )


if __name__ == "__main__":
    main()
