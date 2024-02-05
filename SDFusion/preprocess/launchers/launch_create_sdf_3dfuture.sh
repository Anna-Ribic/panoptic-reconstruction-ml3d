source ./isosurface/LIB_PATH
dset='3dfuture'
# TODO: test which reduce I need
# reduce=2 for 128x128x128, reduce=4 for 64x64x64
reduce=4
# category='all'
# TODO: have to change
category='all'
python3 -u create_sdf.py --dset ${dset} --thread_num 9 --reduce ${reduce} --category ${category}