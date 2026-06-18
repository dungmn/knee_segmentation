import os
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
from cv import segment_baker_cyst

def export_stages_grid(image_path, output_path="logs/tss_dip_stages.png"):
    """
    Chạy thuật toán TSS-DIP ở chế độ debug để sinh ra các ảnh trung gian, 
    sau đó ghép chúng lại thành một biểu đồ dạng lưới (grid) phục vụ báo cáo.
    """
    print(f"Processing image: {image_path}")
    
    # 1. Chạy hàm segment để sinh ra ảnh trung gian trong folder logs/
    # (Đảm bảo cv.py đã bật cờ debug=True để lưu file)
    segment_baker_cyst(image_path, debug=True)
    
    # 2. Đọc lại các ảnh trung gian từ folder logs/
    stage_files = [
        ("logs/0original.png", "1. Anh goc"),
        ("logs/2blurred.png", "2. Khu nhieu (Bilateral)"),
        ("logs/3thresholded.png", "3. Nguong nghiem ngat\n(Detect Stage)"),
        ("logs/4morphological.png", "4. Hinh thai hoc\n(Detect Stage)"),
        ("logs/5seed_mask.png", "5. Hat giong c*"),
        ("logs/6thresh_local.png", "6. Nguong noi long\n(Refine Stage)"),
        ("logs/7morph_local.png", "7. Hinh thai hoc cuc bo\n(Refine Stage)"),
        ("logs/8final_mask.png", "8. Ket qua phan doan")
    ]
    
    images = []
    titles = []
    
    for filepath, title in stage_files:
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            titles.append(title)
        else:
            print(f"Warning: Khong tim thay {filepath}")
            
    if not images:
        print("Khong co anh trung gian nao duoc tao ra.")
        return

    # 3. Tạo plot dạng grid 2 hàng x 4 cột
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Cac giai doan xu ly cua phuong phap TSS-DIP', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i], fontsize=11)
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 4. Lưu ảnh tổng hợp
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDa luu anh tong hop toan bo cac giai doan vao: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TSS-DIP stages grid.")
    parser.add_argument(
        "image", 
        type=str, 
        help="Duong dan den anh can test (vd: data/processed/annotations/post_trans-baker_cyst/batch_000/00000.png)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="logs/tss_dip_stages.png", 
        help="Duong dan luu anh ket qua"
    )
    args = parser.parse_args()
    
    # Đảm bảo thư mục logs tồn tại
    os.makedirs("logs", exist_ok=True)
    
    export_stages_grid(args.image, args.output)
