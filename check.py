import sleap_io as sio

labels = sio.load_file("out2.pkg.slp", open_videos=False)

# Inspect video structure
v = labels.videos[0]
print(f"Type: {type(v)}")
print(f"Backend: {v.backend}")
print(f"Dir: {[a for a in dir(v) if not a.startswith('_')]}")

# Check if shape info is stored elsewhere
if hasattr(v, 'shape'):
    print(f"Shape: {v.shape}")
if hasattr(v, 'img_shape'):
    print(f"img_shape: {v.img_shape}")