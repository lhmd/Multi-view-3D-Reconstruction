colmap feature_extractor \ 
    --database_path output/database.db \
    --image_path output/rgb

colmap exhaustive_matcher \
    --database_path output/database.db

mkdir output/sparse
colmap mapper \
    --database_path output/database.db \
    --image_path output/rgb \
    --output_path output/sparse

mkdir output/cameras
colmap model_converter \
    --input_path output/sparse/0 \
    --output_path output/cameras \
    --output_type TXT

mkdir output/dense
colmap image_undistorter \
    --image_path output/rgb \
    --input_path output/sparse/0 \
    --output_path output/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path output/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true