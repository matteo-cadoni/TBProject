# load vae model
import torch
import matplotlib.pyplot as plt
from vae import ConvVAE

model = ConvVAE()
model.load_state_dict(torch.load('vae_model.pth'))


# generate a random vector of dimension 16
z = torch.randn(1, 16)
# generate an image from the random vector
recon_image = model.generate_image(z)
#squeeze the image
recon_image = recon_image.squeeze(0)
recon_image = recon_image.squeeze(0)

# show the image
plt.imshow(recon_image.detach().cpu().numpy(), cmap='gray')
plt.show()