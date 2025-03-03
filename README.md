# SatPromptedVit
Project for Satellite-based Landuse Classification with Prompted VisionTransformer

# Structure
- job_*.sh files: Training script
- infer.sh: Inference script
- trainer.py: Train + Val code
- evaluator.py: Test code
- model folder: Model code

# Inference
- Put this [model](https://drive.google.com/file/d/1E69YwyhA3N8YJf3Bb80Zpm9KyE_XzsFT/view?usp=sharing) in inference_model folder
- Create folder inference_ some 13-band tif images in inference_tif, put some 13-band tif images there
- Run file infer.sh
