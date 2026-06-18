# Nama : Martha Meslina Florencia
# NPM : 140810230037

import os
import urllib.request
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pylab as plt

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download ImageNet class labels
# ─────────────────────────────────────────────────────────────────────────────
# Buku menyimpan file ini di folder models/imagenet_classes.txt
os.makedirs("models", exist_ok=True)
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
CLASSES_FILE = "models/imagenet_classes.txt"

if not os.path.exists(CLASSES_FILE):
    print("Downloading ImageNet class labels...")
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)

with open(CLASSES_FILE) as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Loaded {len(labels)} ImageNet classes.")  # harusnya 1000


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Fungsi classify() 
# ─────────────────────────────────────────────────────────────────────────────

def classify(img, model_index, model_name, model_pred, labels):
    _, index = torch.max(model_pred, 1)
    model_pred, indices = torch.sort(model_pred, dim=1, descending=True)
    percentage = torch.nn.functional.softmax(model_pred, dim=1)[0] * 100

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except IOError:
        font = ImageFont.load_default(size=30)

    draw.text(
        (5, 5 + model_index * 50),
        '{}, pred: {},{:.2f}%'.format(model_name, labels[index[0]], percentage[0].item()),
        (255, 0, 0),
        font=font
    )

    return indices, percentage


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Transform pipeline — persis dari buku halaman 285
# ─────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 & 5: Load gambar, load model, jalankan inference
# ─────────────────────────────────────────────────────────────────────────────

image_files = ["images/cheetah.jpg", "images/swan.jpg"]

for imgfile in image_files:
    if not os.path.exists(imgfile):
        print(f"[SKIP] File tidak ditemukan: {imgfile}")
        continue

    print(f"\n{'='*50}")
    print(f"Processing: {imgfile}")
    print(f"{'='*50}")

    img = Image.open(imgfile).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    print("Loading VGG19...")
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    vgg19.eval()
    with torch.no_grad():
        pred = vgg19(batch_t)
    classify(img, 0, 'vgg19', pred, labels)
    print(f"  VGG19 done.")

    print("Loading MobileNetV2...")
    mobilenetv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenetv2.eval()
    with torch.no_grad():
        pred = mobilenetv2(batch_t)
    classify(img, 1, 'mobilenetv2', pred, labels)
    print(f"  MobileNetV2 done.")

    print("Loading InceptionV3...")
    inceptionv3 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inceptionv3.eval()
    with torch.no_grad():
        pred = inceptionv3(batch_t)
        if isinstance(pred, tuple):
            pred = pred[0]
    classify(img, 2, 'inceptionv3', pred, labels)
    print(f"  InceptionV3 done.")

    print("Loading ResNet101...")
    resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    resnet101.eval()
    with torch.no_grad():
        pred = resnet101(batch_t)
    indices, percentages = classify(img, 3, 'resnet101', pred, labels)
    print(f"  ResNet101 done.")


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Image Classification: {os.path.basename(imgfile)}", fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("image classified with pytorch")
    axes[0].axis('off')

    top5_labels = [labels[indices[0][i]] for i in range(5)]
    top5_pcts   = [percentages[i].item() for i in range(5)]
    axes[1].bar(top5_labels, top5_pcts, color='steelblue', alpha=0.85)
    axes[1].set_title("Resnet top 5 classes predicted")
    axes[1].set_xlabel("predicted labels")
    axes[1].set_ylabel("predicted percentage")
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()

    out_name = f"output_{os.path.splitext(os.path.basename(imgfile))[0]}.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{out_name}", dpi=150, bbox_inches='tight')
    print(f"  Plot saved: outputs/{out_name}")
    plt.show()

print("\nDone!")