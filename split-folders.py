import splitfolders

input_folder = r"C:\Users\Lenovo\Downloads\archive (3)\brain_tumor_dataset"

splitfolders.ratio(
    input_folder,
    output="brain_tumor_split",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)
