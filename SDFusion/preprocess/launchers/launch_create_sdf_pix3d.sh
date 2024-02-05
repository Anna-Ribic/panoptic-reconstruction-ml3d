source ./isosurface/LIB_PATH
dset='pix3d'
# TODO: test which reduce I need
# reduce=2 for 128x128x128, reduce=4 for 64x64x64
reduce=4
# category='all'
# TODO: have to change
category='chair'
python3 -u create_sdf.py --dset ${dset} --thread_num 9 --reduce ${reduce} --category ${category}