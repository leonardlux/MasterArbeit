import argparse

from tools.combined import generate_new_data_from_config_file 
# This script is getting called by the scrum scripts -> to run stuff on cluster


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process input and output file paths")
    # required
    parser.add_argument("--config", "-c", required=True, help="Path to config file that defines the simulation.")
    parser.add_argument("--output", "-o", required=True, help="Name of the output folder.")
    # optional
    parser.add_argument("--unique_id","-u", help="Some number that makes sure that I do not overwrite datafiles.")

    args = parser.parse_args()

    config_filepath = args.config
    output_folder_name = args.output
    unique_id = args.unique_id 

    print(f"Config File: {config_filepath}")
    print(f"Output Folder: {output_folder_name}")
    generate_new_data_from_config_file(config_filepath, output_folder_name, unique_id)