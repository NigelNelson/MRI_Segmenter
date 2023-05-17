# Help Doc for Hardcoded values
### *Disclaimer*: This is likely not extensive

### Changing Number of Channels:
- Changing the number of input channels likely relates to adding or removing MRI sequences.
- To add or remove MRI sequences:
1. Go to segm/dataprocessing/seg_data_loader.py, in the extract_normal() method, add the sequences you want to *mri_names*
2. Go to segm/model/vit.py, change constructor *channels* param default value to your desired value
3. Go to segm/model/vit.py's *create_vit()* method, change the *default_configs* first arg to your desired number of channels

### Changing Number of Classes:
1. If using UNet, change the *n_classes* param in the constructor call to UNet
2. Change the *n_cls* variable in whichever script you running (ex: segm/train.py)
3. Go to segm/dataprocessing/seg_data_loader.py's *extract_normal()* and make sure that the segmentation masks are being loaded with your desired number of classes
