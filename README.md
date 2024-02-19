1. To run inference please first download our fine-tuned SDFusion model from [link](https://drive.google.com/file/d/1BLG3sJwKfgB2VOIS1VRGI4Ij6W3DrOYE/view?usp=drive_link) and place it under "SDFusion/saved_ckpt/df_steps-12000.pth"
2. Download the panoptic model as mentioned in the main branch and place them in "data/" 
3. Download the front3d dataset and place it under "data/front3d"
4. Run the model using following command from the root folder
´´´bash
  python test_joined_single_image.py -io "<scene_id>/<scene image name without filetype ending>" -sf "df_steps-12000"  
´´
Example:

´´´bash
  python test_joined_single_image.py -io "5d71dabb-9464-4e0c-8d98-e829ade827af/rgb_0034" -sf "df_steps-12000"
´´
