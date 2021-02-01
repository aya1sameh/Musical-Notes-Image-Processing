from difflib import SequenceMatcher , ndiff
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input File")
parser.add_argument("outputfolder", help="Output File")
args = parser.parse_args()
inputDirectory = args.inputfolder
outputDirectory = args.outputfolder

for filename in os.listdir(inputDirectory):   
    try:
        accurate_path = os.path.join(inputDirectory, filename)
        to_check_path = os.path.join(outputDirectory, filename)

        accurate = open(os.path.join(inputDirectory, filename), 'r').read()
        to_check = open(os.path.join(outputDirectory, filename), 'r').read()

        seq = SequenceMatcher(a=accurate, b=to_check)
        diff = ndiff(a=open(accurate_path, 'r').readlines(), b=open(to_check_path, 'r').readlines())

        print("Testing:", os.path.basename(accurate_path),f"Accuracy {seq.ratio() * 100}%")
        print(''.join(diff))
    except:
        pass
