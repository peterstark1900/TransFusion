import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np


class DatasetFishConverter():
    def __init__(self, file_path,file_type):
        self.file_type = file_type
        if self.file_type == 'json':
            self.file_path = file_path
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            # print(self.data['info']['total_frames'])
            self.ori_len = self.data['info']['total_frames']
            self.frames = np.array([])

    def convert_json(self):
        '''
        The shape in each `frame_stamp` is (4,2)
        '''
        for frame_stamp in self.data['frame_stamps']:
            head = np.array([frame_stamp['head'][0],frame_stamp['head'][1],0.0], dtype=np.float16)
            body = np.array([frame_stamp['body'][0],frame_stamp['body'][1],0.0], dtype=np.float16)
            joint = np.array([frame_stamp['joint'][0],frame_stamp['joint'][1],0.0], dtype=np.float16)
            tail = np.array([frame_stamp['tail'][0],frame_stamp['tail'][1],0.0], dtype=np.float16)
            frame = np.vstack((head, body, joint, tail))
            # print(frame)
            # print(frame.shape)
            # add the frame to the frames
            self.frames = np.append(self.frames, frame)
        # reshape the frames
        self.frames = self.frames.reshape(self.ori_len, 4, 3)
        print(self.frames)
        print(self.frames.shape)
        # save the frames into npz format file
        # np.savez(self.file_path[:-5]+'.npz', positions_3d=self.frames)
        np.savez('/home/peter/TransFusion/data/fish-1222-demo19.npz', frames=self.frames)
    
    def convert(self):
        if self.file_type == 'json':
            self.convert_json()

class DatasetFish():
    def __init__(self, file_path, interval_length, stride, batch_size):
        self.file_path = file_path
        # 定义区间长度
        self.interval_length = interval_length
        # 定义滑动步长
        self.stride = stride
        with np.load(self.file_path) as data_file:
            # 通过名称访问数组
            self.frames = data_file['frames']
            
        # print(len(self.frames))
        # print(self.frames.shape)
        # 计算可通过滑动窗口生成的样本数量
        self.num_samples = len(self.frames) - self.interval_length + 1
        self.batch_size = batch_size
        

    def sample(self):    
        """ 生成滑动区间
        每个滑动窗口的形状为
        (self.interval_length, 4, 3)
        """
        num_windows =  (self.frames.shape[0] - self.interval_length) // self.stride + 1
        for i in range(0, num_windows * self.stride, self.stride):
            yield self.frames[i:i + self.interval_length]

    def sampling_generator(self):
        sample_iter = self.sample()
        for i in range(self.num_samples // self.batch_size):
            sample = np.array([])
            for i in range(self.batch_size):
                sample_i = next(sample_iter)
                sample = np.append(sample,sample_i)
            sample = sample.reshape(self.batch_size, self.interval_length, 4, 3)
            yield sample




    
    
def main():
    # file_path = '/home/peter/mmpose/fish-1222-demo19.json'
    # file_type = 'json'
    # dataset = DatasetFishConverter(file_path, file_type)
    # dataset.convert()

    file_path = '/home/peter/TransFusion/data/fish-1222-demo19.npz'
    dataset = DatasetFish(file_path,interval_length=75, stride=1,batch_size=8)
    generator = dataset.sampling_generator()
    temp_data = next(generator)
    # print(temp_data)
    # print(temp_data.shape)
    # temp_data = next(generator)
    # print(temp_data[0])
    # print(temp_data.shape)
    # print(generator)
    



if __name__ == '__main__':
    main()