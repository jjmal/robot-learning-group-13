from cosmos_predict2.data.dataset_video import Dataset

dataset = Dataset(
        dataset_dir = "testdataset",
        num_frames = 30,
        video_size = (1080, 1920),
        is_val = False
    )
