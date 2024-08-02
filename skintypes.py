import jetson_inference
import jetson_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model="skinmodel.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
# print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

print(f"--- This skin is {class_desc}. ---")
if class_desc == "dry":
    print("Here's a skincare routine recommendation!\nMorning: Cleanser-CeraVe Hydrating Cleanser, Moisturizer-CeraVe moisturizing cream, SPF-Round Lab Birch Juice Moisturizing Sun Cream\nNight: Cleanser-CeraVe Hydrating Cleanser, Toners/serums-Olivarrier Dual Moist Toning Lotion, Moisturizer-CeraVe moisturizing cream")
elif class_desc == "normal": 
    print("Here is a skincare routine recommendation!\nMorning: Cleanser-Cetaphil Gentle Skin Cleanser, Moisturizer-Vanicream Daily Facial Moisturizer, SPF-Round Lab Birch Juice Moisturizing Sun Cream\nNight: Cleanser-Cetaphil Gentle Skin Cleanser, Toners/serums-LANEIGE Cream Skin Toner & Moisturizer, Moisturizer-Vanicream Daily Facial Moisturizer") 
elif class_desc == "oily":
    print("Here is a skincare routine recommendation!\nMorning: Cleanser-ANUA Heartleaf Quercetinol Pore Deep CLeansing Foam, Moisturizer-Tatcha The Water Cream, SPF-La Roche Posay Anthelios Clear Skin Sunscreen\nNight: Cleanser-ANUA Heartleaf Quercetinol Pore Deep CLeansing Foam, Toners/serums-Farmacy Deep Sweep Pore Cleaning Toner, Moisturizer-Tatcha The Water Cream")