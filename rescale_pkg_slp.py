# -*- coding: utf-8 -*-
"""
Rescale a SLEAP .pkg.slp file from one resolution to another.

This script does THREE things:
  1. Rescales all point coordinates (labels) proportionally
  2. Resizes all embedded frame images to the new resolution
  3. Updates video dimension metadata in videos_json

Usage:
  python rescale_pkg_slp.py input.pkg.slp output.pkg.slp

Requirements:
  h5py, numpy, opencv-python (all included in SLEAP environment)
"""

import h5py
import json
import numpy as np
import argparse
import shutil
import time

# -- Default configuration --
OLD_WIDTH = 2252
OLD_HEIGHT = 2252
NEW_WIDTH = 3240
NEW_HEIGHT = 2890


def rescale_pkg_slp(input_path, output_path):
    scale_x = NEW_WIDTH / OLD_WIDTH
    scale_y = NEW_HEIGHT / OLD_HEIGHT

    print("=" * 60)
    print("SLEAP pkg.slp Rescaler")
    print("=" * 60)
    print("Input:  " + input_path)
    print("Output: " + output_path)
    print("From:   " + str(OLD_WIDTH) + "x" + str(OLD_HEIGHT))
    print("To:     " + str(NEW_WIDTH) + "x" + str(NEW_HEIGHT))
    print("Scale:  x=" + str(round(scale_x, 6)) + " y=" + str(round(scale_y, 6)))
    print("")

    # -- Step 0: Copy input to output so we can modify in place --
    print("[Step 0] Copying file...")
    shutil.copy2(input_path, output_path)
    print("  Copied to: " + output_path)
    print("")

    with h5py.File(output_path, "r+") as f:
        keys = list(f.keys())
        print("HDF5 top-level keys: " + str(keys))
        print("")

        # =============================================================
        # Step 1: Rescale point coordinates
        # =============================================================
        print("[Step 1] Rescaling point coordinates...")

        # User-labeled points
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

        # Predicted points
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
        # Step 2: Resize embedded frame images
        # =============================================================
        print("[Step 2] Resizing embedded frame images...")

        try:
            import cv2
            print("  Using OpenCV for image resizing")
        except ImportError:
            print("  ERROR: OpenCV not found! Install with:")
            print("    pip install opencv-python-headless")
            print("  Skipping image resize.")
            cv2 = None

        # Find all embedded video groups (video0, video1, etc.)
        video_groups = sorted([k for k in keys if k.startswith("video") and k != "videos_json"])
        print("  Found " + str(len(video_groups)) + " embedded video group(s): " + str(video_groups))

        total_frames_resized = 0

        if cv2 is not None:
            for vg_name in video_groups:
                vg = f[vg_name]

                # Find the video dataset inside the group
                if isinstance(vg, h5py.Group) and "video" in vg:
                    ds_path = vg_name + "/video"
                    ds = vg["video"]
                elif isinstance(vg, h5py.Dataset):
                    ds_path = vg_name
                    ds = vg
                else:
                    sub_keys = list(vg.keys()) if isinstance(vg, h5py.Group) else []
                    print("  " + vg_name + ": skipping (sub-keys: " + str(sub_keys) + ")")
                    continue

                n_frames = ds.shape[0]
                img_format = ds.attrs.get("format", b"png")
                if isinstance(img_format, bytes):
                    img_format = img_format.decode("utf-8")

                print("  " + vg_name + ": " + str(n_frames) + " frames, format=" + img_format)

                # Save all attributes before deleting
                saved_attrs = {}
                for attr_name in ds.attrs:
                    saved_attrs[attr_name] = ds.attrs[attr_name]

                # Also save frame_numbers and source_video if present
                saved_frame_numbers = None
                saved_source_video_attrs = None
                parent_group = ds.parent
                if "frame_numbers" in parent_group:
                    saved_frame_numbers = parent_group["frame_numbers"][:]
                if "source_video" in parent_group:
                    saved_source_video_attrs = {}
                    for attr_name in parent_group["source_video"].attrs:
                        saved_source_video_attrs[attr_name] = parent_group["source_video"].attrs[attr_name]

                # Read all frames, decode, resize, re-encode
                resized_frames = []
                start_time = time.time()

                for frame_i in range(n_frames):
                    raw = ds[frame_i]

                    # Convert to bytes
                    if isinstance(raw, np.ndarray):
                        img_bytes = raw.tobytes()
                    elif isinstance(raw, bytes):
                        img_bytes = raw
                    else:
                        img_bytes = bytes(raw)

                    # Decode
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                    if img is None:
                        print("    WARNING: Could not decode frame " + str(frame_i) + ", keeping original")
                        resized_frames.append(raw)
                        continue

                    # Resize
                    resized = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)

                    # Re-encode
                    if img_format.lower() in ("jpg", "jpeg"):
                        success, encoded = cv2.imencode(".jpg", resized)
                    else:
                        success, encoded = cv2.imencode(".png", resized)

                    if success:
                        resized_frames.append(encoded.tobytes())
                    else:
                        print("    WARNING: Could not encode frame " + str(frame_i) + ", keeping original")
                        resized_frames.append(raw)

                    total_frames_resized += 1

                    if (frame_i + 1) % 10 == 0 or frame_i == n_frames - 1:
                        elapsed = time.time() - start_time
                        fps = (frame_i + 1) / elapsed if elapsed > 0 else 0
                        print("    " + str(frame_i + 1) + "/" + str(n_frames) + " frames (" + str(round(fps, 1)) + " fps)")

                # Delete old dataset and create new variable-length one
                del f[ds_path]

                # Create variable-length byte dataset
                vlen_dt = h5py.special_dtype(vlen=np.uint8)
                new_ds = f.create_dataset(
                    ds_path,
                    shape=(n_frames,),
                    dtype=vlen_dt
                )

                # Write resized frames
                for frame_i in range(n_frames):
                    frame_data = resized_frames[frame_i]
                    if isinstance(frame_data, bytes):
                        new_ds[frame_i] = np.frombuffer(frame_data, dtype=np.uint8)
                    elif isinstance(frame_data, np.ndarray):
                        if frame_data.dtype == np.uint8:
                            new_ds[frame_i] = frame_data
                        else:
                            new_ds[frame_i] = np.frombuffer(frame_data.tobytes(), dtype=np.uint8)
                    else:
                        new_ds[frame_i] = np.frombuffer(bytes(frame_data), dtype=np.uint8)

                # Restore attributes
                for attr_name, attr_val in saved_attrs.items():
                    new_ds.attrs[attr_name] = attr_val

                # Update height/width in attributes if present
                if "height" in new_ds.attrs:
                    new_ds.attrs["height"] = NEW_HEIGHT
                if "width" in new_ds.attrs:
                    new_ds.attrs["width"] = NEW_WIDTH

                print("  " + vg_name + ": done, " + str(n_frames) + " frames resized")

        print("  Total frames resized: " + str(total_frames_resized))
        print("")

        # =============================================================
        # Step 3: Update video metadata in videos_json
        # =============================================================
        print("[Step 3] Updating video metadata...")

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

                # Update backend.shape: [frames, height, width, channels]
                if "backend" in data and "shape" in data["backend"]:
                    old_shape = data["backend"]["shape"]
                    new_shape = list(old_shape)
                    if len(new_shape) >= 3:
                        new_shape[1] = NEW_HEIGHT
                        new_shape[2] = NEW_WIDTH
                    data["backend"]["shape"] = new_shape
                    print("  Video " + str(i) + " (" + short + "): " + str(old_shape) + " -> " + str(new_shape))
                    videos_fixed += 1

                # Also update source_video if present
                if "source_video" in data and data["source_video"] is not None:
                    sv = data["source_video"]
                    if "backend" in sv and "shape" in sv["backend"]:
                        sv_shape = list(sv["backend"]["shape"])
                        if len(sv_shape) >= 3:
                            sv_shape[1] = NEW_HEIGHT
                            sv_shape[2] = NEW_WIDTH
                        sv["backend"]["shape"] = sv_shape

                updated_jsons.append(json.dumps(data))

            # Write back
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
        description="Rescale SLEAP .pkg.slp: coordinates + images + metadata"
    )
    parser.add_argument("input", help="Input .pkg.slp file")
    parser.add_argument("output", help="Output .pkg.slp file")
    parser.add_argument("--old-width", type=int, default=OLD_WIDTH)
    parser.add_argument("--old-height", type=int, default=OLD_HEIGHT)
    parser.add_argument("--new-width", type=int, default=NEW_WIDTH)
    parser.add_argument("--new-height", type=int, default=NEW_HEIGHT)
    args = parser.parse_args()

    OLD_WIDTH = args.old_width
    OLD_HEIGHT = args.old_height
    NEW_WIDTH = args.new_width
    NEW_HEIGHT = args.new_height

    rescale_pkg_slp(args.input, args.output)