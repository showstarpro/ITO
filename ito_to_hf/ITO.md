# How to use tht ITO model

1. cd ./ito_to_hf

2.  run the ``convert_to_hf.py``

``python convert_to_hf.py ViT-B-16 \
 /lpai/models/ito-datacomp1b/itovitbdata1b10ep/ito_datacomp1b_a2_lr5e4/checkpoints/epoch_10.pt \
 ../models_hf/ 
 ``

 3. motify the ``config/llava/llava.jsonc``

``
 "vision_encoder":  models_hf/ito
 "vision_encoder_hidden_size"：768
``

4. Then train the llava-1.5

