import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import argparse
from skimage import exposure # 需要安裝 scikit-image

def apply_advanced_balancing(input_folder, output_folder, template_path=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_paths = sorted([p for ext in extensions for p in glob.glob(os.path.join(input_folder, ext))])

    # 如果沒有指定 template，就用第一張（假設第一張是正常的）
    if template_path is None:
        template_path = image_paths[0]
    
    print(f"使用模板影像進行色彩匹配: {template_path}")
    template_img = cv2.imread(template_path)
    if template_img is None:
        print("錯誤：無法讀取模板影像")
        return

    for img_path in tqdm(image_paths):
        source_img = cv2.imread(img_path)
        if source_img is None: continue

        # 執行直方圖匹配 (Histogram Matching)
        # 這會試圖將 source 的顏色分佈調整得跟 template 一樣
        matched = exposure.match_histograms(source_img, template_img, channel_axis=-1)
        matched = matched.astype(np.uint8)

        # 結合之前的 CLAHE 來增強暗部細節
        lab = cv2.cvtColor(matched, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        final_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, base_name), final_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--template", type=str, default=None, help="正常的陽光照片路徑")
    args = parser.parse_args()

    apply_advanced_balancing(args.input, args.output, args.template)
# import cv2
# import os
# import glob
# from tqdm import tqdm
# import argparse

# def apply_clahe_to_folder(input_folder, output_folder, clip_limit=2.0, tile_grid_size=(8, 8)):
#     # 建立輸出資料夾
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"建立輸出目錄: {output_folder}")

#     # 支援的影像格式
#     extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
#     image_paths = []
#     for ext in extensions:
#         image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

#     if not image_paths:
#         print(f"錯誤：在 {input_folder} 找不到任何影像檔案。")
#         return

#     print(f"找到 {len(image_paths)} 張照片，開始處理 (CLAHE)...")

#     # 初始化 CLAHE
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

#     for img_path in tqdm(image_paths):
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         # LAB 空間處理（保持顏色不失真）
#         lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         cl = clahe.apply(l)
#         limg = cv2.merge((cl, a, b))
#         final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#         # 儲存影像
#         base_name = os.path.basename(img_path)
#         save_path = os.path.join(output_folder, base_name)
#         cv2.imwrite(save_path, final_img)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="影像預處理：應用 CLAHE 平衡光照（針對 VGGT/SfM 優化）")
    
#     # 定義參數
#     parser.add_argument("--input", type=str, required=True, help="原始照片資料夾路徑")
#     parser.add_argument("--output", type=str, required=True, help="輸出預處理照片的路徑")
#     parser.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit (預設 2.0)")
#     parser.add_argument("--grid", type=int, default=8, help="CLAHE tile grid size (預設 8x8)")

#     args = parser.parse_args()

#     # 執行處理
#     apply_clahe_to_folder(
#         args.input, 
#         args.output, 
#         clip_limit=args.clip, 
#         tile_grid_size=(args.grid, args.grid)
#     )
    
#     print("預處理完成！")