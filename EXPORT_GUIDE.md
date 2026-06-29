# 📤 Export Model Guide - Core ML & ONNX

## 🎯 ภาพรวม

แนะนำการแปลง model ทเ่ี ตรนเสรจ็ แลว้ (`.pth`) ไปเปนรปแบบสำหรบั deployment:
- **Core ML (.mlmodel)** - สำหรบั iOS, macOS, watchOS
- **ONNX (.onnx)** - สำหรบั cross-platform (Android, Web, Server)

---

## 📦 การเตรยี มตว

### 1. ตดิ ตงั Dependencies

```bash
# สำหรับ Core ML (macOS เท่านั้น)
pip install coremltools

# สำหรับ ONNX
pip install onnx onnxruntime
```

**หมายเหต:** Core ML export ทำไดบน macOS เทานั้น (แตใชงานไดทกุ platform)

### 2. ตรวจสอบ Model ทเี่ ตรนเสรจ็

```bash
# หา model ล่าสดุ
ls -t /home/user/liveness_trainning/runs/train_v2_*/best_model.pth

# หรอื EfficientNet B3
ls -t /home/user/liveness_trainning/runs/efficientnet_b3_*/best_model.pth
```

---

## 🚀 การ Export Model

### แบบท ี่ 1: Export ทงั Core ML และ ONNX (แนะนำ)

```bash
cd /home/user/liveness_trainning

# Export model ล่าสดุ
./run_export.sh
```

### แบบท ี่ 2: Export Model เฉพาะ

```bash
# MobileNetV3
./run_export.sh \
  --model runs/train_v2_20260627_102528/best_model.pth \
  --model-type mobilenet

# EfficientNet B3
./run_export.sh \
  --model runs/efficientnet_b3_*/best_model.pth \
  --model-type efficientnet
```

### แบบท ี่ 3: Export และทดสอบ Inference

```bash
./run_export.sh \
  --model runs/train_v2_*/best_model.pth \
  --test-inference \
  --test-image /path/to/test.jpg \
  --test-bb /path/to/test_BB.txt
```

### แบบท ี่ 4: Export เฉพาะ Format ทตี่องการ

```bash
# เฉพาะ Core ML
./run_export.sh --coreml-only

# เฉพาะ ONNX
./run_export.sh --onnx-only
```

---

## 📁 Output Files

หลงั export เสรจ็ จะไดไฟลในโฟลเดอรเดยีวกนั กบั model:

```
runs/train_v2_20260627_102528/
├── best_model.pth                      ← Original PyTorch model
├── liveness_mobilenet_224.mlmodel      ← Core ML model
└── liveness_mobilenet_224.onnx         ← ONNX model
```

**สำหรับ EfficientNet B3:**
```
runs/efficientnet_b3_*/
├── best_model.pth                      ← Original PyTorch model
├── liveness_efficientnet_300.mlmodel   ← Core ML model
└── liveness_efficientnet_300.onnx      ← ONNX model
```

---

## 📱 การนำไปใชงาน

### **สญิ่ งสำคญั:** Input ตอง preprocess เหมอื นตอน train!

```
Image + BB → Crop + 30% Padding → Resize → Normalize → Model → Prediction
```

---

## 🍎 iOS/macOS (Core ML)

### 1. เพมิ่ Model เขาโปรเจกต

```swift
// ใน Xcode: File → Add Files to [YourProject]
// เลอื ก liveness_mobilenet_224.mlmodel
```

### 2. สราง Helper Class

```swift
import CoreML
import Vision
import UIKit

class LivenessDetector {
    private let model: VNCoreMLModel
    private let paddingRatio: CGFloat = 0.3
    
    init() throws {
        // Load Core ML model
        let mlModel = try liveness_mobilenet_224(configuration: .init())
        model = try VNCoreMLModel(for: mlModel)
    }
    
    /// ทำนายจากภาพและ bounding box
    /// - Parameters:
    ///   - image: ภาพตนฉบบั (UIImage)
    ///   - boundingBox: Bounding box [x_center, y_center, width, height]
    /// - Returns: (prediction: "LIVE"/"SPOOF", confidence: 0.0-1.0)
    func predict(image: UIImage, boundingBox: [CGFloat]) -> (prediction: String, confidence: CGFloat)? {
        
        // 1. Crop ภาพตาม BB + Padding 30%
        guard let croppedImage = cropWithPadding(
            image: image,
            boundingBox: boundingBox,
            paddingRatio: paddingRatio
        ) else {
            return nil
        }
        
        // 2. Resize เปน 224x224 (หรอื 300x300 สำหรบั EfficientNet)
        guard let resizedImage = resizeImage(image: croppedImage, size: CGSize(width: 224, height: 224)) else {
            return nil
        }
        
        // 3. แปลงเปน CVPixelBuffer
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            return nil
        }
        
        // 4. Run inference
        let request = VNCoreMLRequest(model: model)
        
        do {
            try request.perform(using: [.pixelBuffer(pixelBuffer)])
            
            // 5. ดงึ ผลลพธ
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                let prediction = topResult.identifier == "1" ? "LIVE" : "SPOOF"
                let confidence = CGFloat(topResult.confidence)
                
                return (prediction, confidence)
            }
        } catch {
            print("Error running inference: \(error)")
        }
        
        return nil
    }
    
    /// Crop ภาพตาม BB + Padding
    private func cropWithPadding(image: UIImage, boundingBox: [CGFloat], paddingRatio: CGFloat) -> UIImage? {
        let xCenter = boundingBox[0]
        let yCenter = boundingBox[1]
        let width = boundingBox[2]
        let height = boundingBox[3]
        
        // คำนวณขนาดหลงั เติม padding
        let targetWidth = width * (1 + 2 * paddingRatio)
        let targetHeight = height * (1 + 2 * paddingRatio)
        
        // คำนวณศูนยกลาง
        let centerX = xCenter
        let centerY = yCenter
        
        // คำนวณขอบเขต crop
        let xMin = max(0, centerX - targetWidth / 2)
        let yMin = max(0, centerY - targetHeight / 2)
        let xMax = min(image.size.width, centerX + targetWidth / 2)
        let yMax = min(image.size.height, centerY + targetHeight / 2)
        
        let cropRect = CGRect(x: xMin, y: yMin, width: xMax - xMin, height: yMax - yMin)
        
        // Crop
        guard let cgImage = image.cgImage,
              let croppedCGImage = cgImage.cropping(to: cropRect) else {
            return nil
        }
        
        // Edge padding (จำลองการเตม padding ดวย edge replication)
        let targetRect = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        
        UIGraphicsBeginImageContextWithOptions(targetRect.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }
        
        // วาดภาพท cropped ลงกลาง
        let drawRect = CGRect(
            x: (targetWidth - cropRect.width) / 2,
            y: (targetHeight - cropRect.height) / 2,
            width: cropRect.width,
            height: cropRect.height
        )
        
        UIImage(cgImage: croppedCGImage).draw(in: drawRect)
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    /// Resize ภาพ
    private func resizeImage(image: UIImage, size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: size))
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

// Extension สำหรบั แปลง UIImage เปน CVPixelBuffer
extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(self.size.width),
            Int(self.size.height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let pb = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pb, [])
        let pixelData = CVPixelBufferGetBaseAddress(pb)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: pixelData,
            width: Int(self.size.width),
            height: Int(self.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.translateBy(x: 0, y: self.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(origin: .zero, size: self.size))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pb, [])
        
        return pb
    }
}
```

### 3. ใชงาน

```swift
// 1. สราง detector
let detector = try LivenessDetector()

// 2. เตรียมภาพและ BB (จาก face detector)
let image = UIImage(named: "test")!
let boundingBox: [CGFloat] = [
    x_center,   // จาก face detector
    y_center,
    width,
    height
]

// 3. ทำนาย
if let result = detector.predict(image: image, boundingBox: boundingBox) {
    print("Prediction: \(result.prediction)")
    print("Confidence: \(result.confidence * 100)%")
}
```

---

## 🤖 Android (ONNX)

### 1. เพมิ่ ONNX Runtime

```gradle
// build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

### 2. สราง Helper Class

```kotlin
import android.graphics.*
import org.onnxruntime.*
import java.nio.FloatBuffer

class LivenessDetector(private val context: Context) {
    private val session: OrtSession
    private val paddingRatio = 0.3f
    private val inputSize = 224  // หรือ 300 สำหรับ EfficientNet
    
    init {
        // Load ONNX model จาก assets
        val modelBytes = context.assets.open("liveness_mobilenet_224.onnx").readBytes()
        val env = OrtEnvironment.getEnvironment()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }
    
    data class Prediction(val label: String, val confidence: Float)
    
    fun predict(bitmap: Bitmap, boundingBox: FloatArray): Prediction? {
        // 1. Crop ภาพตาม BB + Padding 30%
        val croppedBitmap = cropWithPadding(bitmap, boundingBox, paddingRatio)
            ?: return null
        
        // 2. Resize เปน 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, inputSize, inputSize, true)
        
        // 3. Normalize
        val inputTensor = preprocess(resizedBitmap)
        
        // 4. Run inference
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs["input"] = OnnxTensor.createTensor(env, inputTensor)
        
        val results = session.run(inputs)
        val output = results[0].value as Array<FloatArray>
        
        // 5. Softmax และดึงผลลพธ
        val probabilities = softmax(output[0])
        val predictedClass = if (probabilities[1] > 0.5) "LIVE" else "SPOOF"
        val confidence = probabilities.maxOrNull() ?: 0f
        
        return Prediction(predictedClass, confidence)
    }
    
    private fun cropWithPadding(bitmap: Bitmap, bb: FloatArray, paddingRatio: Float): Bitmap? {
        val xCenter = bb[0]
        val yCenter = bb[1]
        val width = bb[2]
        val height = bb[3]
        
        val targetWidth = width * (1 + 2 * paddingRatio)
        val targetHeight = height * (1 + 2 * paddingRatio)
        
        val centerX = xCenter
        val centerY = yCenter
        
        val xMin = maxOf(0f, centerX - targetWidth / 2)
        val yMin = maxOf(0f, centerY - targetHeight / 2)
        val xMax = minOf(bitmap.width.toFloat(), centerX + targetWidth / 2)
        val yMax = minOf(bitmap.height.toFloat(), centerY + targetHeight / 2)
        
        val cropRect = RectF(xMin, yMin, xMax, yMax)
        
        // Crop
        val cropped = Bitmap.createBitmap(
            bitmap,
            xMin.toInt(),
            yMin.toInt(),
            (xMax - xMin).toInt(),
            (yMax - yMin).toInt()
        )
        
        // Edge padding (จำลอง)
        val targetBitmap = Bitmap.createBitmap(
            targetWidth.toInt(),
            targetHeight.toInt(),
            Bitmap.Config.ARGB_8888
        )
        
        val canvas = Canvas(targetBitmap)
        canvas.drawColor(Color.GRAY) // Background color
        
        val left = (targetWidth - cropRect.width()) / 2
        val top = (targetHeight - cropRect.height()) / 2
        
        canvas.drawBitmap(cropped, left, top, null)
        
        return targetBitmap
    }
    
    private fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val input = Array(1) { 
            Array(3) { 
                Array(inputSize) { 
                    FloatArray(inputSize) 
                } 
            } 
        }
        
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = bitmap.getPixel(x, y)
                
                input[0][0][y][x] = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                input[0][1][y][x] = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                input[0][2][y][x] = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            }
        }
        
        return input
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = logits.map { Math.exp((it - max).toDouble()).toFloat() }.toFloatArray()
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }
    
    fun close() {
        session.close()
    }
}
```

### 3. ใชงาน

```kotlin
// 1. สราง detector
val detector = LivenessDetector(context)

// 2. เตรียมภาพและ BB
val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test)
val boundingBox = floatArrayOf(x_center, y_center, width, height)

// 3. ทำนาย
val result = detector.predict(bitmap, boundingBox)
println("Prediction: ${result?.label}")
println("Confidence: ${result?.confidence?.times(100)}%")

// 4. ปด session
detector.close()
```

---

## 🌐 Web/Server (ONNX)

### Python

```python
import onnxruntime as ort
from PIL import Image
import numpy as np
from torchvision import transforms

# Load model
session = ort.InferenceSession('liveness_mobilenet_224.onnx')

# Preprocessing function
def preprocess(image, bb, padding_ratio=0.3):
    # Crop with padding
    cropped = crop_with_padding(image, bb, padding_ratio)
    
    # Resize
    resized = cropped.resize((224, 224), Image.LANCZOS)
    
    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(resized).unsqueeze(0).numpy()
    return input_tensor

# Run inference
image = Image.open('test.jpg')
bb = {'x_center': 150, 'y_center': 120, 'width': 80, 'height': 95}

input_tensor = preprocess(image, bb)
output = session.run(None, {'input': input_tensor})[0]

# Get prediction
probabilities = softmax(output[0])
prediction = "LIVE" if probabilities[1] > 0.5 else "SPOOF"
confidence = max(probabilities)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📊 เปรยบเทยบ Formats

| Feature | Core ML | ONNX |
|---------|---------|------|
| **Platform** | iOS, macOS, watchOS | Cross-platform |
| **Performance** | Optimized for Apple | Good on all platforms |
| **Size** | Smaller | Slightly larger |
| **Tools** | Xcode | ONNX Runtime |
| **Quantization** | Supported | Supported |

---

## ⚠️ ขอควรระวง

### 1. Input ตอง Preprocess เหมอื นตอน Train!

```python
# ❌ ผด: สงภาพเตมโดยตรง
output = model(image)  # Accuracy ต่ำ!

# ✅ ถกู ตอง: Crop + Padding กอน
cropped = crop_with_padding(image, bb, 0.3)
resized = cropped.resize((224, 224))
normalized = normalize(resized)
output = model(normalized)
```

### 2. Padding Ratio ตองตรงกนั

- MobileNetV3: `padding_ratio = 0.3`
- EfficientNet B3: `padding_ratio = 0.3`

### 3. Input Size ตองตรงกนั

- MobileNetV3: `224×224`
- EfficientNet B3: `300×300`

---

## 🔧 Troubleshooting

### ปญหา: Core ML export ลมเหลว

**แกไข:**
```bash
# ตรวจสอบวาตดิ ตงั coremltools แลว
pip install coremltools

# ลองใหมดวย CPU
python export_model.py --model model.pth --device cpu
```

### ปญหา: Inference ผลลพธ ผด

**ตรวจสอบ:**
1. Preprocessing ถูกตองหรอื ไม?
2. Padding ratio ตรงกนั หรอื ไม?
3. Input size ตรงกนั หรอื ไม?
4. Normalization mean/std ถูกตองหรอื ไม?

---

## 📞 สรุป

### Export:

```bash
./run_export.sh --model runs/train_v2_*/best_model.pth
```

### ใชงานบน iOS:

```swift
let detector = try LivenessDetector()
let result = detector.predict(image: image, boundingBox: bb)
```

### ใชงานบน Android:

```kotlin
val detector = LivenessDetector(context)
val result = detector.predict(bitmap, boundingBox)
```

### ใชงานบน Python:

```python
session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input': input_tensor})
```

---

**พรอม Deploy แลว! 🚀**
