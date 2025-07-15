## UltraZoom: Generating Gigapixel Images from Regular Photos
#### [Project Page](https://ultra-zoom.github.io) | [Video](https://youtu.be/yIlnyoIxNPI)

This is the official repository for the paper, "UltraZoom: Generating Gigapixel Images from Regular Photos".

<img src="./assets/teaser_v3.gif"/>

### Quickstart

#### 0. Compute Requirement
Training requires an A100 (80GB) or similar/higher. Inference runs on an A40 (40GB) or similar/higher.

#### 1. Installation
```
export ROOT=<path to repository>
conda create -n ultra-zoom python=3.10 -y
conda activate ultra-zoom

# Install diffusers
pip install git+https://github.com/huggingface/diffusers

# Install flux requirements
cd $ROOT/src/diffusers_ultrazoom/examples/dreambooth
pip install -r requirements_flux.txt
pip install wandb prodigyopt lpips datasets scikit-image

# Install dzi conversion repo
cd $ROOT/src
git clone https://github.com/openzoom/deepzoom.py.git
cd deepzoom.py && python setup.py install
pip install -e .

cd $ROOT
```
#### 2. Example Train Command
```
# actual training steps is 250=1000/4 due to gradient accumulation
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export TRAIN_DIR="example_data/pineapple00/closeups_colormatched"
export TEST_PATH="example_data/pineapple00/full.jpg"
export TEST_MASK_PATH="example_data/pineapple00/full_mask.png"
export SCALE_PATH="example_data/pineapple00/scale.txt"
export OUTPUT_DIR="experiments/pineapple00"

accelerate launch src/train.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--closeup_im_dir=$TRAIN_DIR \
--full_im_path=$TEST_PATH \
--full_mask_path=$TEST_MASK_PATH \
--scale_path=$SCALE_PATH \
--output_dir=$OUTPUT_DIR \
--mixed_precision="fp16" \
--instance_prompt="detailed closeup photo of sks texture" \
--validation_prompt="detailed closeup photo of sks texture" \
--resolution=768 \
--train_batch_size=1 \
--guidance_scale=1 \
--gradient_accumulation_steps=4 \
--optimizer="prodigy" \
--learning_rate=1. \
--report_to="wandb" \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_train_epochs=4 \
--validation_epochs=1 \
--seed="0" \
--steps_per_epoch=1000 \
--num_validation_images=3 \
--resume_from_checkpoint=latest \
--extra_downsample=2.0
```
#### 3. Example Inference Command
```
EXP_NAME=pineapple00
EPOCH=checkpoint-ep0001
DATA_NAME=pineapple00

python src/inference.py \
    --exp_name $EXP_NAME \
    --ckpt_name $EPOCH \
    --data_name $DATA_NAME \
    --test_im_path example_data/${DATA_NAME}/full.jpg \
    --test_mask_path example_data/${DATA_NAME}/full_mask.png \
    --closeup_im_dir example_data/${DATA_NAME}/closeups_colormatched \
    --scale_path example_data/${DATA_NAME}/scale.txt \
    --res 768
```
#### 4. Example Train/Inference Output
We provide the trained checkpoint and inference output for the example pineapple capture. Download [here](https://drive.google.com/file/d/1ucCqIuqtb7e2G4sAUsQ2-Ly8ipj6mfHE/view?usp=drive_link).

### Data
We release a preliminary version of the [data](https://drive.google.com/file/d/1BOgFL676ouGMsS_HvbKLfxG4tsXVCOsY/view?usp=drive_link), which includes all necessary files but is not yet organized for direct compatibility with the `UltraZoomDataset` class in `src/train.py`. The release contains the original captures (close-ups, full images, and videos) along with processed outputs such as color-matched images, video point tracking results, and estimated scales (`final_scale.txt`).

A properly organized version will be released soon. In the meantime, this version can be useful if you need to start development right away.

### Release Todos
- [x] Example data/checkpoint, train/inference code
- [x] Preliminary data release
- [ ] Preprocess code & instructions for custom data
- [ ] Paper evaluation code
- [ ] All data/checkpoints
