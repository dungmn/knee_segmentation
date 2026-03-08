import cv2


image_root = "logs/output"
good_images= ["72bb1e5e-f020-11ed-b527-0a580a5f736a_27",
              "72bb4f02-f020-11ed-b527-0a580a5f736a_17",
              "72bb3bf3-f020-11ed-b527-0a580a5f736a_17",
              "72bb4871-f020-11ed-b527-0a580a5f736a_17",
              "72bb4f02-f020-11ed-b527-0a580a5f736a_27",
              "72bb3313-f020-11ed-b527-0a580a5f736a_27",
              "72bb41ed-f020-11ed-b527-0a580a5f736a_17"]

bad_images = []



# vstack images
images = []
for img_name in good_images:
    img_path = f"{image_root}/{img_name}.png"
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
    else:
        print(f"Failed to read image: {img_path}")

if images:
    combined_image = cv2.vconcat(images)
    cv2.imwrite("good_combined.png", combined_image)
    print("Combined image saved as good_combined.png")