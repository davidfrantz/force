import argparse

from synthmix import synthMixCli

parser = argparse.ArgumentParser(description='Force SynthMix...TODO')
parser.add_argument('parameterFile', type=str, help='parameter file')
args = parser.parse_args()
synthMixCli(filenamePrm=args.parameterFile)
