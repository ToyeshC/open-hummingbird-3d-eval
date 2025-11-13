# Image Index
This document provides short descriptions of all images used in the project (paper/poster) for quick reference and reproducibility.

## Images and Notebooks

### [`all_angles_mvimgnet_categories.png`](./all_angles_mvimgnet_categories.png)
A visualization of the MVImgNet categories from all camera angles, used to illustrate the taxonomy and diversity of object viewpoints.

### [`expb_norm.png`](./expb_norm.png) / [`expb_norm.png`](./expb_norm.png)
Plots summarizing Experiment B with normalized and raw metrics. Used in the results section to contrast cross-angle generalization patterns.

### `gt_pred_input_dino_triplet.png`
Triple-panel visualization showing input, ground truth, and DINO prediction.

### [`gt_vs_pred_triceratops.png`](./gt_vs_pred_triceratops.png)
Ground truth vs. predicted segmentation overlay for the Triceratops sample. Shows typical model behavior on textured, irregular geometry.

### [`overlay_gt_pred.png`](./overlay_gt_pred.png)
Composite comparison of ground truth and model prediction for a single input image. Used to illustrate segmentation quality.

### `overlay_gt_pred_dino.png`
Same as overlay_gt_pred.png, but include also the input image without overlays.

### [`hbird_icl_diagram.png`](./hbird_icl_diagram.png)
Diagram of the Hummingbird ICL (in-context learning) setup. Appears in the method section.

### [`small_angles_mvimgnet_categories.png`](./small_angles_mvimgnet_categories.png)
A smaller version of the MVImgNet taxonomy visualization, focusing on just 4 categories and showing visualizations for each angle (0-90).

### [`toy_dragon.png`](./toy_dragon.png)
Example image of the toy dragon object from the dataset at all angles. Used for illustration.

### `image_generation_paper.ipynb`
Notebook used to generate figures included in the paper. 

### `open-hummingbird-eval/examples/hbird_eval_multiview_analysis_memory1024000.ipynb`
The other notebook used to generate figures included in the paper, it is in the `examples` folder.

---

## Per-class Figures (`figures/`)
These are the class-specific plots for Experiment A, each showing viewpoint generalization across difficulty levels.

* `0_background.png` Background class
* `7_stove.png` Stove
* `8_sofa.png` Sofa
* `19_microwave.png` Microwave
* `46_bed.png` Bed
* `57_toy_cat.png` Toy cat
* `60_toy_cow.png` Toy cow
* `70_toy_dragon.png` Toy dragon
* `99_coat_rack.png` Coat rack
* `100_guitar_stand.png` Guitar stand
* `113_ceiling_lamp.png` Ceiling lamp
* `125_toilet.png` Toilet
* `126_sink.png` Sink
* `152_strings.png` Strings
* `166_broccoli.png` Broccoli
* `196_durian.png` Durian

### Combined plots

* `All_Classes.png` mIoU across all classes including background.
* `All_Classes_No_BG.png` Same as above but excluding background.