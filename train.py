import albumentations as A
from unet import Unet
from dataset import CarSegment
from albumentations.pytorch.transforms import ToTensorV2
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision


LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
HEIGHT = 160
WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True

train_dir = 'train'
train_mask_dir = 'train_masks'
test_dir = 'test'
test_mask_dir = 'test_masks'

def train(model, loader, optimizer, loss_fn):
    loop = tqdm(loader)
    model.train()
    for i, (img, mask) in enumerate(loader):
        img = img.to(DEVICE)
        mask = mask.unsqueeze(1).to(DEVICE)
        logits = model(img)
        loss = loss_fn(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def val(model, loader):
    model.eval()
    num_correct = 0
    num_pixel = 0
    dice_score = 0

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.unsqueeze(1).to(DEVICE)
            logits = model(img)
            probas = F.sigmoid(logits)
            preds = (probas > 0.5).float()
            num_correct = (preds == mask).sum()
            num_pixel = torch.numel(mask)
            dice_score += 2*(preds*mask).sum() / ((preds + mask).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixel} with acc {num_correct/num_pixel*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")

def save_predictions_as_imgs(
    loader, model, folder="saved_images/"):
    model.eval()

    for idx, (img,mask) in enumerate(loader):
        img = img.to(device)
        with torch.no_grad():
            preds = F.sigmoid(model(img))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(mask.unsqueeze(1), f"{folder}/masked_{idx}.png")
        torchvision.utils.save_image(img, f"{folder}/masked_{idx}.png")


train_transform = A.Compose(
    [
        A.Resize(height=HEIGHT, width=WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=HEIGHT, width=WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ]
)

model = Unet(3, 1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

train_dataset = CarSegment(train_dir, train_mask_dir, transform=train_transform)
test_dataset = CarSegment(test_dir, test_mask_dir, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    train(model, train_loader, optimizer, loss_fn)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)
    val(model, test_loader)
    save_predictions_as_imgs(test_loader, model)

