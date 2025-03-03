import numpy as np
from scipy import stats
import re
import argparse
from tqdm import tqdm
from scipy.stats import norm, gamma, erlang, expon
import warnings
from scipy.interpolate import interp1d

def eQM_porcentual_delta(model_present, model_future, ref_dataset):
    """
    Remove the biases for each quantile value taking the difference between
    ref_dataset and model_present at each percentile as a kind of systematic bias (delta)
    and add them to model_future at the same percentile.

    returns: downscaled model_present and model_future
    """

    model_present_corrected = np.zeros(model_future.size)

    # Wrap the loop with tqdm for the progress bar
    for ival, model_value in tqdm(enumerate(model_future), total=len(model_future)):
        percentile = stats.percentileofscore(model_present, model_value)
        percentile_ref = np.percentile(ref_dataset, percentile)
        model_present_corrected[ival] = percentile_ref
    return model_present_corrected

def eQM_porcentual_delta_interpolate(model_present, model_future, ref_dataset):
    """
    Smoothly map the model_present distribution to the ref_dataset distribution
    using quantile mapping and interpolation.

    returns: downscaled model_present
    """
    model_present_corrected = np.zeros(model_future.size)

    # Get the unique values and their corresponding quantiles from ref_dataset
    unique_ref_values = np.unique(ref_dataset)
    ref_quantiles = np.array([stats.percentileofscore(ref_dataset, v) for v in unique_ref_values])

    # Create an interpolation function for the ref_dataset quantiles
    interpolation_function = interp1d(ref_quantiles, unique_ref_values, bounds_error=False, fill_value="extrapolate")

    # Map each value in model_present to the corresponding quantile in ref_dataset and interpolate
    for ival, model_value in tqdm(enumerate(model_future), total=len(model_future)):
        #model_percentile = stats.percentileofscore(model_future, model_value)
        #print(f"Before: {model_percentile}")
        model_percentile = stats.percentileofscore(model_present, model_value)
        #print(f"After: {model_percentile}")

        model_present_corrected[ival] = interpolation_function(model_percentile)

    return model_present_corrected


def main():
    parser = argparse.ArgumentParser(description="Quantile normalize scores in a file.")
    parser.add_argument("sourcefile", help="Input data file name")
    parser.add_argument("targetfile", help="Target data file name")
    args = parser.parse_args()

    # To hold the modified lines
    new_lines = []
    new_lines2 = []

    try:
        # Extract data2 from the target file
        data2 = []
        with open(args.targetfile, 'r', encoding='utf-8') as file:
            for line in file:
                parts = re.split(r'<ENDSENTENCE>', line)
                if len(parts) >= 3:
                    try:
                        score = float(parts[-1].strip())
                        data2.append(score)
                    except ValueError:
                        continue

        # Ensure data2 is a numpy array
        data2 = np.array(data2)
        print(max(data2))
        print(min(data2))

        data3 = []
        with open('formatted_result.txt', 'r', encoding='utf-8') as file:
            for line in file:
                parts = re.split(r'<ENDSENTENCE>', line)
                if len(parts) >= 3:
                    try:
                        score = float(parts[-1].strip())
                        data3.append(score)
                    except ValueError:
                        continue

        # Ensure data3 is a numpy array
        data3 = np.array(data3)


        # Open the source file and read the data
        with open(args.sourcefile, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Extract data1 from the lines
        data1 = []
        for line in lines:
            parts = re.split(r'<ENDSENTENCE>', line)
            if len(parts) != 3:
                continue
            try:
                score = float(parts[-1].strip())
                data1.append(score)
            except ValueError:
                continue

        # Ensure data1 is a numpy array
        data1 = np.array(data1)

        # Perform quantile normalization
        normalized_data1 = eQM_porcentual_delta(data3, data1, data2)
        normalized_data2 = eQM_porcentual_delta_interpolate(data3, data1, data2)
        # Create a dictionary to map original data1 values to normalized values
        normalization_dict = {orig: norm for orig, norm in zip(data1, normalized_data1)}
        normalization_dict2 = {orig: norm for orig, norm in zip(data1, normalized_data2)}
        # Now, process each line to transform the scores
        for line in lines:
            # Split the line by the <ENDSENTENCE> symbol
            parts = re.split(r'<ENDSENTENCE>', line)
            if len(parts) != 3:
                print(f"Skipping malformed line: {line}")
                continue

            str1, str2, score = parts
            score = score.strip()

            # Transform the score using the normalization dictionary
            try:
                z_score = float(score)
                new_score = normalization_dict[z_score]

                # Create the new line
                new_line = f"{str1}<ENDSENTENCE>{str2}<ENDSENTENCE>{new_score}\n"
                new_lines.append(new_line)
            except ValueError:
                print(f"Invalid score value: {score}")

            try:
                z_score = float(score)
                new_score = normalization_dict2[z_score]

                # Create the new line
                new_line = f"{str1}<ENDSENTENCE>{str2}<ENDSENTENCE>{new_score}\n"
                new_lines2.append(new_line)
            except ValueError:
                print(f"Invalid score value: {score}")

        # Save the target file hwith the modified scores
        with open("da-qm.txt", 'w', encoding='utf-8') as file:
            file.writelines(new_lines)
        with open("da-qm-interpolate.txt", 'w', encoding='utf-8') as file:
            file.writelines(new_lines2)

        print(f"File '{args.targetfile}' has been updated successfully.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")

if __name__ == "__main__":
    main()
