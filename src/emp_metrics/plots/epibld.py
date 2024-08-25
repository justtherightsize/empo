import json
import argparse


def load_json_file(filename):
    """
    Load a JSON file and return its contents.
    :param filename: The name of the JSON file to load.
    :return: The contents of the JSON file as a Python dictionary.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def main(name):
    """
    Main method to load and print a JSON file based on the provided filename.
    :param name: The name of the JSON file to load.
    """
    data = load_json_file(name)
    if data is not None:
        der = float(data["epitome"]["diff_ER"])
        dex = float(data["epitome"]["diff_EX"])
        dip = float(data["epitome"]["diff_IP"])
        ad = (der + dex + dip) / 3
        print("{:.3f}".format(ad))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a JSON file.")
    parser.add_argument("-n", '--name', type=str,
                        help="The name of the JSON file to load.")
    args = parser.parse_args()
    main(args.name)
