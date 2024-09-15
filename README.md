# BoFiT

This repository contains raw data and pose data extracted using visual models for the BoFiT (Body Fitness Training) benchmark. The dataset is designed to facilitate research and development in fitness and pose estimation.

### Repository Structure
`videos/`: Contains 2360 videos used in the benchmark.
`clips_info.json`: A JSON file that provides metadata for each video, including:
- Action name
- Action type
- Targeted muscle
- Required equipment
- Detailed action instruction
- Cover image URL
- Video URL
`inc_angle_representation/`: Contains the absolute joint coordinates extracted by the visual model, with each file named after its corresponding video.
`tb_angle_representation/`: Contains Tait-Bryan angles for joint poses extracted by the visual model, with each file named after its corresponding video.