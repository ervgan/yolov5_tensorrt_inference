import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Convert .pt file to .wts")
        self.parser.add_argument(
            "-w",
            "--weights",
            required=True,
            help="Input weights (.pt) file path (required)",
        )
        self.parser.add_argument(
            "-o", "--output", help="Output (.wts) file path (optional)"
        )

    def parse_args(self):
        self.args = self.parser.parse_args()
        if not os.path.isfile(self.args.weights):
            raise SystemExit("Invalid input file")
        if not self.args.output:
            self.args.output = os.path.splitext(self.args.weights)[0] + ".wts"
        elif os.path.isdir(self.args.output):
            self.args.output = os.path.join(
                self.args.output,
                os.path.splitext(os.path.basename(self.args.weights))[0] + ".wts",
            )
        return self.args.weights, self.args.output


# Class will convert a pytorch model into a wts format
class WTSConverter:
    def __init__(self, pt_file, wts_file):
        self.pt_file = pt_file
        self.wts_file = wts_file
        self.device = select_device("cpu")
        # Load FP32 weights
        self.model = torch.load(self.pt_file, map_location=self.device)

    def load_model(self):
        print(f"Loading {self.pt_file}")
        self.model = self.model["ema" if self.model.get("ema") else "model"].float()
        # update anchor_grid info
        anchor_grid = (
            self.model.model[-1].anchors * self.model.model[-1].stride[..., None, None]
        )
        delattr(self.model.model[-1], "anchor_grid")  # model.model[-1] is detect layer
        # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
        self.model.model[-1].register_buffer("anchor_grid", anchor_grid)
        self.model.model[-1].register_buffer("strides", self.model.model[-1].stride)
        self.model.to(self.device).eval()

    def write_wts_file(self):
        print(f"Writing into {self.wts_file}")
        with open(self.wts_file, "w") as file:
            file.write("{}\n".format(len(self.model.state_dict().keys())))

            for param, tensor in self.model.state_dict().items():
                np_array = tensor.reshape(-1).cpu().numpy()
                file.write("{} {} ".format(param, len(np_array)))
                for weight in np_array:
                    file.write(" ")
                    file.write(struct.pack(">f", float(weight)).hex())
                file.write("\n")


def main():
    parser = Parser()
    args = parser.parse_args()
    pt_file, output_file = args
    wts_converter = WTSConverter()
    print(f"Generating .wts for detection model")
    wts_converter.load_model(pt_file, output_file)
    wts_converter.write_wts_file()
