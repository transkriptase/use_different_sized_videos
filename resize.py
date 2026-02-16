import sleap_io as sio

labels = sio.load_file("out2.pkg.slp", open_videos=False)

# Fix video references
all_videos = set(lf.video for lf in labels.labeled_frames)
for v in all_videos:
    if v not in labels.videos:
        labels.videos.append(v)

# Restore resized dimensions for all videos
for v in labels.videos:
    print(f"Before: {v.filename} -> {v.shape}")
    v.shape = [v.shape[0], 3240, 2890, v.shape[3]]  # [frames, height, width, channels]
    print(f"After:  {v.filename} -> {v.shape}")

sio.save_file(labels, "resized_pkg_fixed.slp")
print("Done!")