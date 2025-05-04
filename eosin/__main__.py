import argparse
import pandas as pd
from eosin.parser import Parser as PDFParser

if __name__ == "__main__":
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)

    args_parser = argparse.ArgumentParser(description="Bank Statement Parser CLI")
    args_parser.add_argument("file", help="File to parse")
    args = args_parser.parse_args()

    parser = PDFParser(args.file)
    print(parser.parse())