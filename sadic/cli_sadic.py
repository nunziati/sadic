#!/usr/bin/env python3

import argparse
import importlib

from sadic import sadic


def load_dictionary(file_path, dictionary_name):
    try:
        module = importlib.import_module(file_path)
        my_dict = getattr(module, dictionary_name)
        return my_dict
    except (ImportError, AttributeError) as e:
        print(f"Error loading dictionary from {file_path}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Sadic')

    parser.add_argument('input', help='The input argument, that can either be the path to a .pdb or .tar.gz file containing a PDB format file of a protein, or a string representing a PDB protein code.')
    parser.add_argument('-f', '-F', '--file', help='The input argument is a path to a .pdb or .tar.gz file containing a PDB format file of a protein.')
    parser.add_argument('-c', '-C', '--code', help='The input argument is a string representing a PDB protein code.')

    parser.add_argument('--config', help='The configuration file to use. If not specified, the default configuration is used.')
    parser.add_argument('-o', '-O', '--output', help='Path of a .pdb or .tar.gz file to save the result.')

    args = parser.parse_args()

    input_argument = None

    if args.input is not None and not any([args.file, args.code]):
        input_argument = {'mode': 'infer', 'arg': args.input}
    elif args.file is not None and not any([args.input, args.code]):
        if args.file.endswith('.pdb'):
            input_argument = {'mode': 'pdb', 'arg': args.file}
        elif args.file.endswith('.tar.gz'):
            input_argument = {'mode': 'gz', 'arg': args.file}
        else:
            parser.error('The file must be a .pdb or .tar.gz file.')
    elif args.code is not None and not any([args.input, args.file]):
        input_argument = {'mode': 'code', 'arg': args.code}

    if input_argument is None:
        parser.error('Exactly one of argument, -f (or -F, or --file), or -c (or -C, or --code) must be specified.')

    return args, input_argument


def main():
    args, input_argument = parse_args()

    input_arg = input_argument['arg']
    input_mode = input_argument['mode']
    sadic_config = load_dictionary(args.config, 'sadic_config') if args.config is not None else {}
    output_config = load_dictionary(args.config, 'output_config') if args.config is not None else {}
    output_argument = args.output

    result = sadic(input_arg, input_mode=input_mode, **sadic_config)

    if output_argument is not None:
        result.save_pdb(output_argument, **output_config)


if __name__ == '__main__':
    main()