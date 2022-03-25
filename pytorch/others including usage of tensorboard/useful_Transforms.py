from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
writer = SummaryWriter("logs")
img = Image.open('imge/21.png')

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor)
writer.add_image("totensor",img_tensor)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.3,0.3,0.3], [0.5,0.5,0.5])
img_norm = trans_norm(img_tensor) #input[channel]= (input[channel] - output[channel])/std[channel]
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,3)


trans_size = transforms.Resize((512,512))
img_size = trans_size(img) #PIL -> PIL
img_resize = trans_totensor(img_size) #PIL - TOTENSOR
writer.add_image("Resize",img_resize,0)

trans_resize =transforms.Resize(512)
trans_compose =transforms.Compose([trans_resize,trans_totensor])
img_resize2 = trans_compose(img)
writer.add_image("Resize",img_resize2,1)

#random crop
trans_crop = transforms.RandomCrop(512)
trans_compose2 = transforms.Compose([trans_crop, trans_totensor])
for i in range(10):
    img_Crop = trans_compose2(img)
    writer.add_image("random", img_Crop, i)











writer.close()

