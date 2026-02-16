# -*- coding: utf-8 -*-
"""
Rescale a regular SLEAP .slp file from one resolution to another.

This script does TWO things:
  1. Rescales all point coordinates (labels) proportionally
  2. Updates video dimension metadata in videos_json

(No image processing needed - regular .slp files don't have embedded frames)

Usage:
  python rescale_slp.py input.slp output.slp
  python rescale_slp.py handlabels_S2_3.2_N1_pos.slp handlabels_S2_3.2_N1_+_rescaled_output.slp
"""

import h5py
import json
import numpy as np
import argparse
import shutil

# -- Default configuration --
OLD_WIDTH = 2252
OLD_HEIGHT = 2252
NEW_WIDTH = 3240
NEW_HEIGHT = 2890


def rescale_slp(input_path, output_path):
    scale_x = NEW_WIDTH / OLD_WIDTH
    scale_y = NEW_HEIGHT / OLD_HEIGHT

    print("=" * 60)
    print("SLEAP .slp Rescaler")
    print("=" * 60)
    print("Input:  " + input_path)
    print("Output: " + output_path)
    print("From:   " + str(OLD_WIDTH) + "x" + str(OLD_HEIGHT))
    print("To:     " + str(NEW_WIDTH) + "x" + str(NEW_HEIGHT))
    print("Scale:  x=" + str(round(scale_x, 6)) + " y=" + str(round(scale_y, 6)))
    print("")

    # Copy input to output
    print("[Step 0] Copying file...")
    shutil.copy2(input_path, output_path)
    print("")

    with h5py.File(output_path, "r+") as f:

        # =============================================================
        # Step 1: Rescale point coordinates
        # =============================================================
        print("[Step 1] Rescaling point coordinates...")

        if "points" in f:
            pts = f["points"]
            data = pts[:]
            n_points = len(data)
            count = 0
            for i in range(n_points):
                x_val = data[i]["x"]
                y_val = data[i]["y"]
                if not (np.isnan(x_val) or np.isnan(y_val)):
                    data[i]["x"] = x_val * scale_x
                    data[i]["y"] = y_val * scale_y
                    count += 1
            pts[...] = data
            print("  User points: " + str(count) + " / " + str(n_points) + " rescaled")
        else:
            print("  No 'points' dataset found")

        if "pred_points" in f:
            pred_pts = f["pred_points"]
            data = pred_pts[:]
            n_points = len(data)
            count = 0
            for i in range(n_points):
                x_val = data[i]["x"]
                y_val = data[i]["y"]
                if not (np.isnan(x_val) or np.isnan(y_val)):
                    data[i]["x"] = x_val * scale_x
                    data[i]["y"] = y_val * scale_y
                    count += 1
            pred_pts[...] = data
            print("  Pred points: " + str(count) + " / " + str(n_points) + " rescaled")
        else:
            print("  No 'pred_points' dataset found")

        print("")

        # =============================================================
        # Step 2: Update video metadata in videos_json
        # =============================================================
        print("[Step 2] Updating video metadata...")

        if "videos_json" in f:
            raw = f["videos_json"]
            updated_jsons = []
            videos_fixed = 0

            for i in range(len(raw)):
                entry = raw[i]
                if isinstance(entry, bytes):
                    entry = entry.decode("utf-8")

                data = json.loads(entry)
                filename = data.get("filename", "?")
                short = filename.split("/")[-1] if "/" in filename else filename

                if "backend" in data and "shape" in data["backend"]:
                    old_shape = data["backend"]["shape"]
                    new_shape = list(old_shape)
                    if len(new_shape) >= 3:
                        new_shape[1] = NEW_HEIGHT
                        new_shape[2] = NEW_WIDTH
                    data["backend"]["shape"] = new_shape
                    print("  Video " + str(i) + " (" + short + "): " + str(old_shape) + " -> " + str(new_shape))
                    videos_fixed += 1

                if "source_video" in data and data["source_video"] is not None:
                    sv = data["source_video"]
                    if "backend" in sv and "shape" in sv["backend"]:
                        sv_shape = list(sv["backend"]["shape"])
                        if len(sv_shape) >= 3:
                            sv_shape[1] = NEW_HEIGHT
                            sv_shape[2] = NEW_WIDTH
                        sv["backend"]["shape"] = sv_shape

                updated_jsons.append(json.dumps(data))

            del f["videos_json"]
            encoded = [s.encode("utf-8") for s in updated_jsons]
            f.create_dataset("videos_json", data=encoded, maxshape=(None,))
            print("  Updated " + str(videos_fixed) + " video metadata entries")
        else:
            print("  WARNING: No videos_json found")

    print("")
    print("=" * 60)
    print("DONE!")
    print("Output: " + output_path)
    print("New resolution: " + str(NEW_WIDTH) + "x" + str(NEW_HEIGHT))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rescale SLEAP .slp: coordinates + metadata"
    )
    parser.add_argument("input", help="Input .slp file")
    parser.add_argument("output", help="Output .slp file")
    parser.add_argument("--old-width", type=int, default=OLD_WIDTH)
    parser.add_argument("--old-height", type=int, default=OLD_HEIGHT)
    parser.add_argument("--new-width", type=int, default=NEW_WIDTH)
    parser.add_argument("--new-height", type=int, default=NEW_HEIGHT)
    args = parser.parse_args()

    OLD_WIDTH = args.old_width
    OLD_HEIGHT = args.old_height
    NEW_WIDTH = args.new_width
    NEW_HEIGHT = args.new_height

    rescale_slp(args.input, args.output)
