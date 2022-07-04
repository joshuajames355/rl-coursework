import torchvision,torch

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Grayscale(1),
    torchvision.transforms.Resize((84,84)),
    torchvision.transforms.ToTensor(),
])

def preprocess(image):
    return transforms(image)