"""TODO: Add docstring."""

import argparse
import ast

# Create an empty csv file with header in the current directory if file does not exist
import csv
import os
import time
from io import BytesIO

import cv2
import librosa
import numpy as np
import pyarrow as pa
import requests
from dora import Node
from PIL import Image

CAT_URL = "https://i.ytimg.com/vi/fzzjgBAaWZw/hqdefault.jpg"


def get_cat_image():
    """
    Get a cat image as a numpy array.

    :return: Cat image as a numpy array.
    """
    # Fetch the image from the URL
    response = requests.get(CAT_URL)
    response.raise_for_status()

    # Open the image using PIL

    image = Image.open(BytesIO(response.content))
    # Convert the image to a numpy array

    image_array = np.array(image)
    cv2.resize(image_array, (640, 480))
    # Convert RGB to BGR for

    return image_array


AUDIO_URL = "https://github.com/dora-rs/dora-rs.github.io/raw/refs/heads/main/static/Voicy_C3PO%20-Don't%20follow%20me.mp3"


def get_c3po_audio():
    """
    Download the C-3PO audio and load it into a NumPy array using librosa.
    """

    temp_audio_file = os.getenv("AUDIO", "temp_audio.wav")

    # Check if temp file exist
    if not os.path.exists(temp_audio_file):
        # Download the audio file
        response = requests.get(AUDIO_URL)
        if response.status_code != 200:
            raise Exception(
                f"Failed to download audio file. Status code: {response.status_code}"
            )

        # Save the audio file temporarily
        with open(temp_audio_file, "wb") as f:
            f.write(response.content)

    # Load the audio file into a NumPy array using librosa
    audio_data, sample_rate = librosa.load(temp_audio_file, sr=None)

    return audio_data, sample_rate


def write_to_csv(filename, header, row):
    """
    Create a CSV file with a header if it does not exist, and write a row to it.
    If the file exists, append the row to the file.

    :param filename: Name of the CSV file.
    :param header: List of column names to use as the header.
    :param row: List of data to write as a row in the CSV file.
    """
    file_exists = os.path.exists(filename)

    with open(
        filename, mode="a" if file_exists else "w", newline="", encoding="utf8"
    ) as file:
        writer = csv.writer(file)

        # Write the header if the file is being created
        if not file_exists:
            writer.writerow(header)
            print(f"File '{filename}' created with header: {header}")

        # Write the row
        writer.writerow(row)
        print(f"Row written to '{filename}': {row}")


def main():
    # Handle dynamic nodes, ask for the name of the node in the dataflow, and the same values as the ENV variables.
    """TODO: Add docstring."""
    parser = argparse.ArgumentParser(description="Simple arrow sender")

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="pyarrow-sender",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Arrow Data as string.",
        default=None,
    )

    args = parser.parse_args()

    audio, sample_rate = get_c3po_audio()
    node = Node(
        args.name,
    )  # provide the name to connect to the dataflow if dynamic node
    name = node.dataflow_descriptor()["nodes"][1]["path"]

    durations = {}
    speed = {}
    for _ in range(10):
        node.send_output(
            "audio",
            pa.array(audio.ravel()),
            {"sample_rate": sample_rate},
        )
        start_time = time.time()
        for _ in range(3):
            event = node.next()
            if event is not None and event["type"] == "INPUT":
                id = event["id"]

                duration = time.time() - start_time
                print("got event: ", event)
                print("duration: ", duration)
                tokens = event["metadata"].get("tokens", 6)
                if id in durations:
                    durations[id].append(duration)
                else:
                    durations[id] = [duration]
        time.sleep(3)
    for id, duration in durations.items():
        duration = np.array(duration)

        print(
            f"\nAverage duration: {sum(duration) / len(duration)}"
            + f"\nMax duration: {max(durations)}"
            + f"\nMin duration: {min(duration)}"
            + f"\nMedian duration: {np.median(duration)}"
            + f"\nMedian frequency: {1 / np.median(duration)}"
        )
        write_to_csv(
            f"benchmark_{id}.csv",
            [
                "path",
                "date",
                "average_duration(s)",
                "max_duration(s)",
                "min_duration(s)",
                "median_duration(s)",
                "median_frequency(Hz)",
                "average_speed(tok/s)",
                "total_tokens",
            ],
            [
                id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                sum(duration) / len(duration),
                max(duration),
                min(duration),
                np.median(duration),
                1 / np.median(duration),
                tokens,
            ],
        )


if __name__ == "__main__":
    main()
