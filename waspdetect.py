#!/usr/bin/env python3
"""
Folder Wasp Detection Test
Usage: python test_folder.py path/to/folder
Runs inference on all images in folder and creates side-by-side comparison images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Better for local display
import time
import os
import sys
from pathlib import Path  # Added this import
from ultralytics import YOLO

def test_folder(folder_path, confidence_threshold=0.3):
    """Test both models on all images in a folder and create comparison images"""
    
    print("🐝 FOLDER WASP DETECTION TEST")
    print("=" * 50)
    print(f"📁 Testing folder: {folder_path}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print()
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Load models
    base_model = None
    fine_tuned_model = None
    
    print("🔄 Loading models...")
    
    # Load base model
    if os.path.exists("best.pt"):
        try:
            base_model = YOLO("best.pt")
            print("✅ Base model loaded: best.pt")
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
    else:
        print("❌ best.pt not found")
    
    # Load fine-tuned model
    if os.path.exists("finetuned_best.pt"):
        try:
            fine_tuned_model = YOLO("finetuned_best.pt")
            print("✅ Fine-tuned model loaded: finetuned_best.pt")
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
    else:
        print("❌ finetuned_best.pt not found")
    
    if not base_model and not fine_tuned_model:
        print("❌ No models found! Make sure you have best.pt and/or finetuned_best.pt")
        return
    
    # Find all images in folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"📸 Found {len(image_files)} images")
    
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving results to: {output_dir}/")
    print()
    
    # Process each image
    base_total_detections = 0
    ft_total_detections = 0
    base_total_time = 0
    ft_total_time = 0
    successful_tests = 0
    
    for i, image_path in enumerate(image_files):
        print(f"📷 Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ❌ Could not load image: {image_path.name}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Test base model
            base_detections = 0
            base_time = 0
            
            if base_model:
                try:
                    start_time = time.time()
                    base_results = base_model(str(image_path), conf=confidence_threshold, verbose=False)
                    base_time = time.time() - start_time
                    
                    base_img = base_results[0].plot()
                    base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                    
                    base_detections = len(base_results[0].boxes) if base_results[0].boxes is not None else 0
                    base_total_detections += base_detections
                    base_total_time += base_time
                    
                    axes[0].imshow(base_img_rgb)
                    axes[0].set_title(f"Base Model\n{base_detections} detections | {base_time*1000:.0f}ms", fontsize=14)
                    axes[0].axis('off')
                    
                except Exception as e:
                    print(f"  ❌ Base model error: {e}")
                    axes[0].imshow(img_rgb)
                    axes[0].set_title("Base Model\nERROR", fontsize=14)
                    axes[0].axis('off')
            else:
                axes[0].imshow(img_rgb)
                axes[0].set_title("Base Model\nNOT LOADED", fontsize=14)
                axes[0].axis('off')
            
            # Test fine-tuned model
            ft_detections = 0
            ft_time = 0
            
            if fine_tuned_model:
                try:
                    start_time = time.time()
                    ft_results = fine_tuned_model(str(image_path), conf=confidence_threshold, verbose=False)
                    ft_time = time.time() - start_time
                    
                    ft_img = ft_results[0].plot()
                    ft_img_rgb = cv2.cvtColor(ft_img, cv2.COLOR_BGR2RGB)
                    
                    ft_detections = len(ft_results[0].boxes) if ft_results[0].boxes is not None else 0
                    ft_total_detections += ft_detections
                    ft_total_time += ft_time
                    
                    axes[1].imshow(ft_img_rgb)
                    axes[1].set_title(f"Fine-tuned Model\n{ft_detections} detections | {ft_time*1000:.0f}ms", fontsize=14)
                    axes[1].axis('off')
                    
                except Exception as e:
                    print(f"  ❌ Fine-tuned model error: {e}")
                    axes[1].imshow(img_rgb)
                    axes[1].set_title("Fine-tuned Model\nERROR", fontsize=14)
                    axes[1].axis('off')
            else:
                axes[1].imshow(img_rgb)
                axes[1].set_title("Fine-tuned Model\nNOT LOADED", fontsize=14)
                axes[1].axis('off')
            
            # Add main title
            fig.suptitle(f"{image_path.name} | Confidence: {confidence_threshold}", fontsize=16, y=0.98)
            
            plt.tight_layout()
            
            # Save comparison
            output_name = f"{output_dir}/comparison_{image_path.stem}.png"
            plt.savefig(output_name, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Base: {base_detections} detections, Fine-tuned: {ft_detections} detections")
            print(f"  💾 Saved: {output_name}")
            
            successful_tests += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
    
    # Summary
    print(f"\n📊 FOLDER SUMMARY")
    print("=" * 30)
    print(f"📸 Images processed: {successful_tests}/{len(image_files)}")
    
    if successful_tests > 0:
        if base_model:
            print(f"🎯 Base Model:")
            print(f"  Total detections: {base_total_detections}")
            print(f"  Avg per image: {base_total_detections/successful_tests:.1f}")
            print(f"  Avg time: {base_total_time/successful_tests*1000:.1f}ms")
        
        if fine_tuned_model:
            print(f"🎯 Fine-tuned Model:")
            print(f"  Total detections: {ft_total_detections}")
            print(f"  Avg per image: {ft_total_detections/successful_tests:.1f}")
            print(f"  Avg time: {ft_total_time/successful_tests*1000:.1f}ms")
        
        # Winner
        if base_model and fine_tuned_model:
            print(f"\n🏆 FOLDER WINNER:")
            if base_total_detections > ft_total_detections:
                print(f"🥇 Base model detected {base_total_detections - ft_total_detections} more wasps total")
            elif ft_total_detections > base_total_detections:
                print(f"🥇 Fine-tuned model detected {ft_total_detections - base_total_detections} more wasps total")
            else:
                print(f"🤝 Both models detected the same number of wasps")
    
    print(f"\n💾 All comparison images saved in: {output_dir}/")
    print(f"✅ Folder test completed!")

def main():
    """Main function with command line argument handling"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("🐝 FOLDER WASP DETECTION TEST")
        print("=" * 40)
        print("Usage:")
        print(f"  python {sys.argv[0]} path/to/folder")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} yellowjacket")
        print(f"  python {sys.argv[0]} deployment_dataset\\test\\images")
        print(f"  python {sys.argv[0]} C:\\Users\\Name\\Pictures")
        print()
        print("Creates side-by-side comparison images for all images in folder")
        print("Saves results in 'comparison_results/' directory")
        return
    
    folder_path = sys.argv[1]
    
    # Test the folder
    test_folder(folder_path, confidence_threshold=0.3)

if __name__ == "__main__":
    main()