from ultralytics import SAM

# Load a model
model = SAM("sam2.1_l.pt")

# Display model information (optional)
model.info()
results = model("data/marvin-scaled.png",
    device="cuda",
)
results[0].show()
