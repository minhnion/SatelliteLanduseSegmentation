# SatPromptedVit
Project for Satellite-based Landuse Classification with Prompted VisionTransformer

# Structure
- `job_*.sh` files: Training script
- `infer.sh`: Inference script
- `trainer.py`: Train + Val code
- `evaluator.py`: Test code
- `model folder`: Model code

# Inference
- Create folder `inference_model`
- Put this [model](https://drive.google.com/file/d/1DrT-25VjLgSV5aj6LWZUHJNKK2JA2yIr/view?usp=drive_link) in `inference_model` folder
- Create folder `inference_tif`, put some 13-band tif images in `inference_tif`
- Run file `infer.sh`
