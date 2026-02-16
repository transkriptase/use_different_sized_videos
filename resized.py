import sleap_io as sio

# Load the resized file
labels = sio.load_file("out2.pkg.slp", open_videos=False)

print(f"Videos: {len(labels.videos)}")
print(f"Labeled frames: {len(labels.labeled_frames)}")

# Check if video references are valid
for i, lf in enumerate(labels.labeled_frames):
    if lf.video not in labels.videos:
        print(f"Broken frame {i}: frame_idx={lf.frame_idx}")

# Fix: make sure all videos referenced by frames are in the video list
all_videos = set(lf.video for lf in labels.labeled_frames)
for v in all_videos:
    if v not in labels.videos:
        labels.videos.append(v)
        print(f"Added missing video: {v}")

# Save fixed version
sio.save_file(labels, "resized_pkg_fixed.slp")
print("Saved fixed file!")