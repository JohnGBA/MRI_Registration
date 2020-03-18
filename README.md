# MRI_Registration
Random forest algorithm to predict rotation and translation applied to an MRI slice image.


In order to propose an accurate diagnosis during the longitudinal monitoring of the same region in magnetic resonance imaging, it is important that the same acquisition volumes chosen in week 0 and week X are the same to allow the study of evolution.
To carry out an MRI acquisition of the brain, the operator must first define the region of interest to be visualized, as well as the orientation of the cutting planes. To do this, it first performs low-resolution acquisitions through fast MRI sessions. This step is done manually. However, because of inter- and intra-operator variability, differences in positioning and orientation occur from one week to the next, making the diagnosis delicate or false. Based on the location images, the operator schedules a complete and detailed scan of the brain. This complete acquisition takes much longer. Therefore, it is essential that the positioning made with the location images is correct in order to obtain comparable acquisitions for longitudinal monitoring, and not having to relaunch an exam again.
