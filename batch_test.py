import os
import shutil

TB_DIR = "dataset/val/tb"
NORMAL_DIR = "dataset/val/normal"

OUTPUT_ROOT = "batch_results"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# First 10 TB images
tb_images = sorted(os.listdir(TB_DIR))[:10]

# First 10 Healthy images
normal_images = sorted(os.listdir(NORMAL_DIR))[:10]


def run_group(image_list, image_dir, label):

    save_dir = os.path.join(
        OUTPUT_ROOT,
        label
    )

    os.makedirs(save_dir, exist_ok=True)

    for img_name in image_list:

        img_path = os.path.join(
            image_dir,
            img_name
        )

        print("\nTesting:", img_path)

        os.system(
            f'python3 predict_tb.py "{img_path}"'
        )

        # Copy generated files

        if os.path.exists(
            "outputs/heatmap_overlay.png"
        ):
            shutil.copy(
                "outputs/heatmap_overlay.png",
                os.path.join(
                    save_dir,
                    img_name.replace(
                        ".png",
                        "_heatmap.png"
                    )
                )
            )

        if os.path.exists(
            "outputs/tb_hotspots.png"
        ):
            shutil.copy(
                "outputs/tb_hotspots.png",
                os.path.join(
                    save_dir,
                    img_name.replace(
                        ".png",
                        "_hotspot.png"
                    )
                )
            )


run_group(
    tb_images,
    TB_DIR,
    "tb"
)

run_group(
    normal_images,
    NORMAL_DIR,
    "normal"
)

print("\nDONE")