import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_mask(img, pred_mask, gt_mask, output_path="inference_result.png"):
    # class definition
    classes = [
        {"name": "background",   "id": 0, "color": "#000000"},
        {"name": "fat",          "id": 1, "color": "#fafa37"},
        {"name": "tendon",       "id": 2, "color": "#fa7dbb"},
        {"name": "muscle",       "id": 3, "color": "#ff0000"},
        {"name": "femur",        "id": 4, "color": "#cccfd2"},
        {"name": "artery",       "id": 5, "color": "#2a7dd1"},
        {"name": "bakers_cyst",  "id": 6, "color": "#aaf0d1"},
    ]

    # initialize RGB mask
    rgb_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    rgb_gt_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

    # fill colors
    for c in classes:
        rgb = (np.array(mcolors.to_rgb(c["color"])) * 255).astype(np.uint8)
        rgb_gt_mask[gt_mask == c["id"]] = rgb
        rgb_pred_mask[pred_mask == c["id"]] = rgb
        
    # add legend
    # import matplotlib.patches as mpatches
    # handles = [mpatches.Patch(color=c["color"], label=c["name"]) for c in classes]
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plot
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(rgb_pred_mask); plt.title("Predicted Mask"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(rgb_gt_mask); plt.title("Ground Truth Mask"); plt.axis("off")
    plt.show()


    plt.savefig(output_path+".pdf", format="pdf")
    print(f"[blue]Inference result saved to:[/blue] {output_path}.pdf")