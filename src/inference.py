### 1. Imports
import os
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse
import time
import psutil  
from torchvision import transforms
import deepzoom

from diffusers import FluxControlNetModel

# custom diffusers code
from diffusers_ultrazoom.pipelines.flux.pipeline_flux_controlnet_ultrazoom_inference import FluxControlNetMultiDiffusionPipeline

global_start_time = time.time()



### 2. Helper functions
def pil_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),                   # Convert to tensor in range [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5),   # Normalize to range [-1, 1]
                             (0.5, 0.5, 0.5))
    ])
    tensor = transform(img).unsqueeze(0)        # Add batch dimension (1, 3, H, W)
    return tensor



### 3. Load params
parser = argparse.ArgumentParser()

# paths
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--ckpt_name", type=str, required=True)
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--test_im_path", type=str, required=True)
parser.add_argument("--test_mask_path", type=str, required=True)
parser.add_argument("--closeup_im_dir", type=str, required=True)
parser.add_argument("--scale_path", type=str, required=True)

# inference params
parser.add_argument("--res", type=int, default=1024)  # should be consistent with train resolution
parser.add_argument("--num_inference_steps", type=int, default=28)
parser.add_argument("--chunk_size", type=int, default=2500)  # number of patches to pass through vae each time
parser.add_argument("--stride_method", type=str, default="linear", choices=["constant", "linear"])
parser.add_argument("--seed", type=int, default=0)

# other params
parser.add_argument("--delete_prev", action="store_true")
parser.add_argument("--remask", action="store_true")  # skip inference, just remask the output

# debug params
parser.add_argument("--debug", action="store_true")  # debug mode, run inference on a small region
parser.add_argument("--debug_size", type=int, default=2048)  # debug size, in pixels
parser.add_argument("--verbose", action="store_true")  # verbose mode, print more runtime, memory usage

args = parser.parse_args()



### 4. params and paths
data_name = args.data_name
test_im_path = args.test_im_path
lora_path = f"experiments/{args.exp_name}/{args.ckpt_name}"
num_inference_steps = args.num_inference_steps
prompt = "detailed closeup photo of sks texture"
res = args.res
np.random.seed(args.seed)

# set stride
if args.stride_method == "constant":
  stride_list = np.array([res//2] * num_inference_steps)
elif args.stride_method == "linear":  # best one
  stride_list = np.round(np.linspace(res//2, res//8*7, num_inference_steps)/16).astype(int)*16

# load scale
scale = float(open(args.scale_path, 'r').readlines()[0].strip())

# out dir
out_dir = os.path.join(lora_path, f"results")
if args.debug:
  out_dir = out_dir + f"_debug{args.debug_size}"
if args.delete_prev:
  os.system(f"rm -rf {out_dir}")
os.makedirs(out_dir, exist_ok=True)

# out path and intermediate dir
out_path = os.path.join(out_dir, f"scale{scale}_nsteps{num_inference_steps}_out.png")
intermediate_dir = out_path.replace("_out.png", "_intermediate")  # resume inference process from intermediate results
os.makedirs(intermediate_dir, exist_ok=True) 

# save args
with open(os.path.join(out_dir, "args_inference.txt"), "w") as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}: {value}\n")



### 5. get inference region and padding params
full_mask = Image.open(args.test_mask_path).convert("L")
full_mask_np = np.array(full_mask)
full_w, full_h = full_mask.size

# pad so that the image is divisible by 16
full_upscaled_w, full_upscaled_h = int(round(full_w*scale)), int(round(full_h*scale))
full_upscaled_w_pad = int(np.ceil(full_upscaled_w/16) * 16) - full_upscaled_w
full_upscaled_h_pad = int(np.ceil(full_upscaled_h/16) * 16) - full_upscaled_h

# extra padding in all directions
top_pad, bottom_pad, left_pad, right_pad = res, res, res, res

# make sure the crop coordinates are divisible by 16
full_ys, full_xs = np.where(full_mask_np == 255)
xs_min, xs_max = int(np.floor(full_xs.min() * scale / 16) * 16), int(np.ceil((full_xs.max()+1) * scale / 16) * 16)
ys_min, ys_max = int(np.floor(full_ys.min() * scale / 16) * 16), int(np.ceil((full_ys.max()+1) * scale / 16) * 16)

# shift all coords by padding
xs_min += left_pad
xs_max += left_pad
ys_min += top_pad
ys_max += top_pad

# get debug crop, centered around crop center
if args.debug:
  assert args.debug_size % 16 == 0, f"debug_size must be divisible by 16, got {args.debug_size}"
  xs_min = (xs_min + xs_max)//2
  xs_max = xs_min + args.debug_size
  ys_min = (ys_min + ys_max)//2
  ys_max = ys_min + args.debug_size

# extra padding in all directions
xs_min -= res
xs_max += res
ys_min -= res
ys_max += res

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

if not args.remask:

  ### 6. create list of coords
  coords_list = []  # (h_starts, w_starts) for all time steps
  label_list = []  # 0 for padding, 1 for valid
  num_windows = []
  for i in range(num_inference_steps):

    stride = stride_list[i]
    h_starts = list(range(ys_min, ys_max-res+1, stride))
    if h_starts[-1] + res < ys_max:
        h_starts.append(ys_max - res)
    w_starts = list(range(xs_min, xs_max-res+1, stride))
    if w_starts[-1] + res < xs_max:
        w_starts.append(xs_max - res)

    coords_list_t = []
    label_list_t = []
    for i, h_start in enumerate(h_starts):
      for j, w_start in enumerate(w_starts):
        coords_list_t.append((h_start, w_start))
        label_list_t.append(0 if i <= 1 or j <= 1 or i >= len(h_starts)-2 or j >= len(w_starts)-2 else 1)
    coords_list.append(np.array(coords_list_t))
    label_list.append(np.array(label_list_t))
    num_windows.append(len(h_starts) * len(w_starts))

  print(f"[LOG] Average num_windows per time step: {np.mean(num_windows):.2f}")
  print(f"[LOG] Total num_windows: {np.sum(num_windows)}")
  print(f"[LOG] Total estimated run time on A100 (lower than actual): {np.sum(num_windows)*18/num_inference_steps/60/60:.2f} hours")


  ### 7. preview inference regions
  inference_vis = np.zeros_like(full_mask_np)
  for (h, w), label in zip(coords_list[0], label_list[0]):
    if label == 1:
      lr_w, lr_h = int(round((w-res)/scale)), int(round((h-res)/scale))
      lr_res = int(round(res/scale))
      inference_vis[lr_h:lr_h+lr_res, lr_w:lr_w+lr_res] = 1
  inference_vis_pil = Image.fromarray((inference_vis*255).astype('uint8'))
  inference_vis_pil.save(os.path.join(out_dir, "inference_region_preview.png"))


  ### 8. upscale control image
  start_time = time.time()
  full_im = Image.open(test_im_path)
  control_im = full_im.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
  control_im = add_margin(control_im, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, (255, 255, 255))
  control_w, control_h = control_im.size
  if args.verbose:
    print(f"[LOG] took {(time.time() - start_time):.2f} seconds to upscale control image from {full_w}x{full_h} to {control_w}x{control_h}")
    print(f"[LOG] After upscale control image, memory usage: {psutil.Process().memory_info().rss / 1e9} GB")
  control_im_copy = control_im.copy()


  ### 9. load pipeline
  start_time = time.time()
  controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=torch.bfloat16
  )
  pipe = FluxControlNetMultiDiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
  ).to("cuda")
  print(f"[LOG] Loading lora weights from {lora_path}")
  pipe.load_lora_weights(lora_path)
  if args.verbose:
    print(f"[LOG] Time taken to load full model: {time.time() - start_time} seconds")
    print(f"[LOG] After loading full model, memory usage: {psutil.Process().memory_info().rss / 1e9} GB")  


  ### 10. multi-diffusion inference
  with torch.no_grad():
    image, intermediate_list = pipe(
        prompt=prompt, 
        control_image=control_im,
        coords_list=coords_list,
        label_list=label_list,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=num_inference_steps, 
        guidance_scale=3.5,
        height=control_h,
        width=control_w,
        h_win=res,
        w_win=res,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        intermediate_dir=intermediate_dir,
        return_dict=False,
        debug_size=args.debug_size,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

else:  # remask the output; skipping multi-diffusion inference
  start_time = time.time()
  image = Image.open(out_path)
  image = add_margin(image, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, (255, 255, 255))
  if args.verbose:
    print(f"[LOG] Output image loaded in {(time.time() - start_time):.2f} seconds")

  # upscale control image
  start_time = time.time()
  full_im = Image.open(test_im_path)
  control_im = full_im.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
  control_im_copy = add_margin(control_im, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, (255, 255, 255))
  if args.verbose:
    print(f"[LOG] Control image loaded in {(time.time() - start_time):.2f} seconds")


### 11. upscale mask, convert to binary, mask out pixels, crop out padding
start_time = time.time()
control_mask = full_mask.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
control_mask = Image.fromarray((np.array(control_mask) >= 128).astype(np.uint8) * 255)
control_mask = add_margin(control_mask, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, 0)
control_mask = control_mask.convert('L').convert('1')
image = Image.composite(image, control_im_copy, control_mask)

# take out previous padding
image = image.crop((left_pad, top_pad, full_upscaled_w+left_pad, full_upscaled_h+top_pad))
if args.verbose:
  print(f"[LOG] took {(time.time() - start_time):.2f} seconds to apply mask and unpad")


### 12. save the entire image
start_time = time.time()
image.save(out_path)
if args.verbose:
  print(f"[LOG] took {(time.time() - start_time):.2f} seconds to save the output image, size: {image.size}")
start_time = time.time()
image.resize((full_w, full_h), resample=Image.BICUBIC).save(out_path.replace(".png", "_preview.png"))
if args.verbose:
  print(f"[LOG] took {(time.time() - start_time):.2f} seconds to resize and save the preview image")


### 13. create the html file
dzi_dir = intermediate_dir.replace("_intermediate", "_dzi")
os.makedirs(dzi_dir, exist_ok=True)
html_path = os.path.join(dzi_dir, "index.html")
closeup_im_path = os.path.join(args.closeup_im_dir, sorted(os.listdir(args.closeup_im_dir))[0])
closeup_im_rel_path = closeup_im_path.split('example_data/')[-1]
with open(html_path, "w") as f:
  f.write(f"""<!DOCTYPE html>
<html>
  <head>
      <title>{data_name}</title>
      <script src="../../../../../src/openseadragon/openseadragon.min.js"></script>
      <style>
          .viewer-container {{
              display: flex;
              width: 100%;
              height: 100vh;
          }}
          .viewer {{
              width: 50%;
              height: 100%;
          }}
          img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
          }}
      </style>
  </head>
  <body>
      <div class="viewer-container">
          <div id="viewer1" class="viewer"></div>
          <div id="viewer2" class="viewer"></div>
      </div>
      
      <script type='text/javascript'>
          // Initialize both viewers
          var viewer1 = OpenSeadragon({{
              id: "viewer1",
              prefixUrl: "../../../../../src/openseadragon/images/",
              tileSources: {{
                type: 'image',
                url: '../../../../../example_data/{closeup_im_rel_path}'
              }},
              showNavigationControl: true
          }});
          var viewer2 = OpenSeadragon({{
              id: "viewer2",
              prefixUrl: "../../../../../src/openseadragon/images/",
              tileSources: './image.dzi',
              showNavigationControl: true
          }});
          viewer1.setControlsEnabled(false);
          viewer2.setControlsEnabled(false);

          var label1 = document.createElement('div');
          label1.innerHTML = "Closeup (captured, {scale:.3f}x)";
          label1.style.position = "absolute";
          label1.style.top = "10px";
          label1.style.right = "10px";
          label1.style.background = "green";
          label1.style.color = "white";
          label1.style.padding = "6px";
          label1.style.fontSize = "24px";
          label1.style.zIndex = 1000;
          viewer1.container.appendChild(label1);


          var label2 = document.createElement('div');
          label2.innerHTML = "Ours";
          label2.style.position = "absolute";
          label2.style.top = "10px";
          label2.style.right = "10px";
          label2.style.background = "green";
          label2.style.color = "white";
          label2.style.padding = "6px";
          label2.style.fontSize = "24px";
          label2.style.zIndex = 1000;
          viewer2.container.appendChild(label2);

      </script>
      
  </body>
</html>
  """)

### 14. convert output to dzi
creator = deepzoom.ImageCreator(
    tile_size=128,
    tile_overlap=2,
    tile_format="png",
    image_quality=0.8,
    resize_filter="bicubic",
)
creator.create(out_path, os.path.join(dzi_dir, "image.dzi"))


print("--------------------------------")
print(f"Total time taken: {(time.time() - global_start_time):.2f} seconds")
print(f"[LOG] Results saved to {out_path}")