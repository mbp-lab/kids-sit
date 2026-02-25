Video-based Social Interaction Behavior Analysis with the Simulated Interaction Task for Children (Kids-SIT)

All analysis steps are available in the notebooks/ directory.

0️⃣ Feature Extraction

Extract features using:

OpenFace

PyAFAR

REHG Eye Contact CNN

All extracted features are merged into a unified dataset.

1️⃣ Preprocessing

Keep only relevant video segments for analysis

Add metadata (e.g., speaker identity and segment information)

2️⃣ Annotation

Integrate human annotator labels

Create a behavior_agreed column (instances where all annotators agree)

3️⃣ Behavior Extraction

Convert raw features into interpretable behavioral features, for example:

Gaze deviation from gaze_angle_x and gaze_angle_y

Nodding behavior from head pitch and yaw

4️⃣ Threshold Selection

Fine-tune threshold values for:

Smile detection

Gaze detection

Thresholds are selected to maximize agreement with human annotations.

5️⃣ Subjective Impression

Compute results related to subjective impressions.

6️⃣ Verbal Responses

Analyze and report results for verbal responses.

7️⃣ Non-Verbal Responses

Analyze and report results for non-verbal responses.

8️⃣ Classification

Evaluate clinical applicability of Kid-SIT by classifying participants diagnosed with SAD vs. non-SAD.

⚙️ Setup

To run the code and notebooks:

1️⃣ Create the Conda Environment (Python 3.9)
conda env create -f environment.yml
conda activate <environment-name>
2️⃣ Launch Jupyter
jupyter notebook