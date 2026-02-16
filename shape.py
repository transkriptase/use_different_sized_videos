import sleap_io as sio

labels = sio.load_file("resized_pkg.slp", open_videos=False)

# Fix video references
all_videos = set(lf.video for lf in labels.labeled_frames)
for v in all_videos:
    if v not in labels.videos:
        labels.videos.append(v)

# Restore resized dimensions
for v in labels.videos:
    print(f"Before: {v.backend_metadata['shape']}")
    v.backend_metadata["shape"] = [v.backend_metadata["shape"][0], 3240, 2890, v.backend_metadata["shape"][3]]
    print(f"After:  {v.backend_metadata['shape']}")

sio.save_file(labels, "resized_pkg_fixed.slp")
print("Done!")