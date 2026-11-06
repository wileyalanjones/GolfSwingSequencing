from golfswing import utils as ut
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE  

class GolfDataset:
    """
    Dataset Object for Random Sampling of GolfDB Videos
    """
    def __init__(
            self,
            dataframe,
            vid_dir,
            clip_len=64,
            num_events=8,
            train=True,
            repeat_per_video=2
            ):
       
        self.id_to_events = ut.get_events_dataframe(dataframe, num_events)
        self.vid_dir = vid_dir
        self.clip_len = clip_len
        self.num_events = num_events
        self.repeat_per_video = repeat_per_video
        self.train = train
    
        self.records = []
        for vid in self.id_to_events.keys():
            vid_path = f"{self.vid_dir}/{vid}.mp4"
            num_frames = ut.get_num_frames(vid_path)
            self.records.append({"id": vid, 
                                 "vid_path": vid_path,
                                 "num_frames": num_frames, 
                                 "events": self.id_to_events[vid]})
        
    def __len__(self):
        return len(self.records) * self.repeat_per_video
        
    def gen(self):
        """
        Yields Random Segment of Video for Dataset
        """
        for rec in self.records:
            for _ in range(self.repeat_per_video):
                if self.train:
                    idxs = ut.get_random_clip_indexes(
                        rec['num_frames'], 
                        self.clip_len,
                        rec["events"])
                else:
                    idxs = ut.get_random_clip_indexes(
                        rec['num_frames'], 
                        self.clip_len)
                
                frames = ut.get_frames_by_idxs(rec["vid_path"], idxs)
                labels = ut.get_clip_labels(idxs, rec['events'])

                yield frames, labels
                
def make_tf_ds(dataset, batch_size=2, shuffle=True, 
               augment=True, num_classes=9, clip_len=None):
    """
    Creates Dataset Tensor for Keras Model
    """
        
    if not clip_len:
        clip_len = dataset.clip_len
    
    output_sig = (
        tf.TensorSpec(shape=(clip_len, 160, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(clip_len,), dtype=tf.int32),
        )
    
    ds = tf.data.Dataset.from_generator(dataset.gen, output_signature=output_sig)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=8*batch_size, reshuffle_each_iteration=True)

    # Augmentation for more robust training    
    aug = tf.keras.Sequential([ 
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(.05),
        tf.keras.layers.RandomRotation(.05),
        ]) 
    
    def augment_frames(x, y):
        x = preprocess_input(x) # prepare frames for MobileNet
        if augment:
            x = aug(x)
        return x, y
    
    ds = ds.map(augment_frames, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat() 
    ds = ds.prefetch(AUTOTUNE)
    
    return ds 


class FullVideoGolfDataset:
    """
    Dataset Object for Full Video Validation, PCE 
    """
    def __init__(self, dataframe, vid_dir, num_events=8):
        self.id_to_events = ut.get_events_dataframe(dataframe, num_events)
        self.vid_dir = vid_dir
        self.num_events = num_events
        
        self.records = []
        for vid in self.id_to_events.keys():
            vid_path = f"{self.vid_dir}/{vid}.mp4"
            num_frames = ut.get_num_frames(vid_path)
            self.records.append({
                "id": vid,
                "vid_path": vid_path,
                "num_frames": num_frames,
                "events": self.id_to_events[vid]
            })

    def __len__(self):
        return len(self.records)

    def gen(self):
        """
        Yields full video frames and labels
        """
        for rec in self.records:
            T = rec["num_frames"]

            idxs = np.arange(T, dtype=np.int32)

            frames = ut.get_frames_by_idxs(rec["vid_path"], idxs)   # (T,H,W,C)
            labels = ut.get_clip_labels(idxs, rec["events"])        # (T,)

            frames = frames.astype(np.float32)
            labels = labels.astype(np.int32)

            yield frames, labels


def make_full_video_tf_ds(full_dataset):
    """
    Tensor Creator for full video validation
    """
    
    # No Batching since clips are not uniform length
    output_sig = ( 
        tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )  

    ds = tf.data.Dataset.from_generator(full_dataset.gen, output_signature=output_sig)

    def preprocess(frames, labels):
        frames = preprocess_input(frames)
        return frames, labels

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)
