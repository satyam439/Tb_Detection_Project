import cv2
import numpy as np

img = cv2.imread(
    "outputs/heatmap_overlay_tb.png"
)

if img is None:
    raise Exception(
        "Could not load outputs/heatmap_overlay_tb.png"
    )

red = (
    (img[:,:,2] > 150) &
    (img[:,:,1] < 100) &
    (img[:,:,0] < 100)
)

ys, xs = np.where(red)

if len(xs) == 0:
    raise Exception("No hotspot found")

cx = int(xs.mean())
cy = int(ys.mean())

tb_burden = round(
    100 * len(xs) / (224*224),
    2
)

np.save(
    "outputs/tb_center.npy",
    np.array([cx,cy])
)

with open(
    "outputs/tb_stats.txt",
    "w"
) as f:

    f.write(f"center_x={cx}\n")
    f.write(f"center_y={cy}\n")
    f.write(f"tb_burden={tb_burden}\n")

print("TB hotspot:",cx,cy)
print("TB burden:",tb_burden,"%")