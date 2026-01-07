This is a tool to process instagram images for Lora training, its features are:

    - Gradio interface.
    - Blip Image Captioning.
    - Automatic resizing to 512x512 for Lora training.
    - Automatic centering to face (as center as possible).
    - Caption manual modification interface.
    - Automatic folder creation with images and captions in txt for Lora.
    - Only can scrap public instagram accounts.

I really enjoyed the interpretation of Ivana Baquero in AltaMar series, so lets make an example with her instagram public account. The interface has two tabs, the first one is for download, resize and crop the images, meanwhile the second one is for Blip captioning and prompting correction.

![alt text](https://github.com/makiJanus/instagram-lora-tool/blob/main/git_images/instagram_tools.png?raw=true)
![alt text](https://github.com/makiJanus/instagram-lora-tool/blob/main/git_images/pre_training_tools.png?raw=true)

## Installing on Windows + AMD Strix Halo

source: https://github.com/Comfy-Org/ComfyUI?tab=readme-ov-file#installing

You can proceed in a similar fashion to the ComfyUI tutorial

> RDNA 3.5 (Strix halo/Ryzen AI Max+ 365):
```
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```