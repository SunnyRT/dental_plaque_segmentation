import os
import shutil


def reorganize_ply_files(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of patient folders and sort them
    patient_folders = sorted(os.listdir(input_dir))
    
    
    # Iterate over each patient folder
    for patient_folder in patient_folders:
        patient_path = os.path.join(input_dir, patient_folder)
        
        if os.path.isdir(patient_path):  # Check if it's a directory
            patient_index = patient_folder.split('.')[0]
            patient_index = patient_index.zfill(2)  # Pad with zeros
            for file_name in os.listdir(patient_path):
                # Determine if it's a LowerJaw or UpperJaw scan
                if "LowerJawScan" in file_name:
                    new_file_name = f"00{patient_index}01.ply"
                elif "UpperJawScan" in file_name:
                    new_file_name = f"00{patient_index}02.ply"
                else:
                    continue  # Skip files that are not PLYs
                
                # Copy and rename the file to the output directory
                src_file = os.path.join(patient_path, file_name)
                dst_file = os.path.join(output_dir, new_file_name)
                shutil.copyfile(src_file, dst_file)
            

input_dir = 'D:/sunny/Codes/DPS/data_raw/unlabelled'
output_dir = 'D:/sunny/Codes/DPS/data_raw/origin'


# Reorganize and rename the PLY files
reorganize_ply_files(input_dir, output_dir)

print("Reorganization and renaming completed.")