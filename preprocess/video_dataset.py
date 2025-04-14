import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from preprocess import video_transforms
from preprocess import volume_transforms

class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, static=False, args=None):
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.args = args
        self.static = static
        
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        # annotation path, first column is filename, second column is label
        
        classes, class_to_idx = find_classes(self.data_path)
        samples = make_dataset(self.data_path, class_to_idx, extensions=('.avi', '.mp4'))
        
        self.dataset_samples = [sample[0] for sample in samples]
        self.label_array = [sample[1] for sample in samples]

        # train, validation, test transform
        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif (mode == 'supervised'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.RandomCrop(size=(self.crop_size, self.crop_size)),
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == "supervised":
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            # input is filename, loadvideo return the frame images, based on frame sample rate and segments(between frame, defaut 1)
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    # warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            # use _aug_frame for augmentation of frames, if num_sample > 1, return num_sample aug clips for the same video
            if self.num_segment <= 1:
                if self.mode == "train":
                    buffer = self._aug_frame(buffer, args)
                else:
                    buffer = self.data_transform(buffer)
            else:
                # frame_list = []
                # label_list = []
                # index_list = []
                # for i in range(self.num_segment):
                #     new_frames = self._aug_frame(buffer, args)
                #     label = self.label_array[index]
                #     frame_list.append(new_frames)
                #     label_list.append(label)
                #     index_list.append(index)
                frame_list = []
                label_list = []
                index_list = []
                for indi_buffer in buffer:
                    if self.mode == "train":
                        new_frames = self._aug_frame(indi_buffer, args)
                    else:
                        new_frames = self.data_transform(indi_buffer)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                    
                return frame_list, label_list

            return buffer, self.label_array[index]

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    # warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            # return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]
            return buffer, self.label_array[index]
        
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.2, 0.766],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=self.new_height,
            max_scale=self.new_width,
            crop_size=self.crop_size,
            random_horizontal_flip=True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            # init VideoReader can optimize with threads and ctx
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []
        
        if self.mode == "train" or self.mode == "supervised":
            num_segment = self.num_segment
        else:
            num_segment = 1
        # num_segment = 1

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        # seg_len = len(vr) // num_segment

        # all_index = []
        # for i in range(num_segment):
        #     if seg_len <= converted_len:
        #         index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
        #         index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
        #         index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        #     else:
        #         end_idx = np.random.randint(converted_len, seg_len)
        #         str_idx = end_idx - converted_len
        #         index = np.linspace(str_idx, end_idx, num=self.clip_len)
        #         index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
        #         if self.static:
        #             rand_index = np.random.randint(0,len(index))
        #             index = [index[rand_index] + i*seg_len for j in range(8)]
        #         else:
        #             index = list(index + i*seg_len)
        #     all_index.extend(list(index))

        # all_index = all_index[::int(sample_rate_scale)]
        # vr.seek(0)
        # buffer = vr.get_batch(all_index).asnumpy()
        available_index = list(range(0,len(vr) - converted_len))
        if len(available_index) < num_segment:
            return []
        np.random.shuffle(available_index)
        if num_segment == 1:
            start_idx = available_index.pop()
            end_idx = start_idx + converted_len - 1
            index = np.linspace(start_idx, end_idx, self.clip_len)
            vr.seek(0)
            batch = vr.get_batch(index).asnumpy()
            return batch

        indexes = []
        for i in range(num_segment):
            start_idx = available_index.pop()
            end_idx = start_idx + converted_len - 1
            index = np.linspace(start_idx, end_idx, self.clip_len)
            indexes.append(index)
        indexes.sort(key=lambda x:x[0])
        buffers = []
        for index in indexes:
            buffer = vr.get_batch(index).asnumpy()
            buffers.append(buffer)
            vr.seek(0)
        return buffers

    def __len__(self):
        return len(self.dataset_samples)

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    color_jitter=False,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
        if color_jitter:
            frames = video_transforms.color_jitter(frames, 0.6, 0.6, 0.6)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

