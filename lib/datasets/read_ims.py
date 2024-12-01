import numpy as np
import h5py
import tifffile as tif

class Ims_Image():
    '''
    ims image: [z,y,x]
    input roi and returned image: [z,y,x]
    '''
    def __init__(self,ims_path,channel=0):
        self.hdf = h5py.File(ims_path,'r')
        level_keys = list(self.hdf['DataSet'].keys())
        channel_keys = [key for key in self.hdf['DataSet'][level_keys[0]]['TimePoint 0']]
        self.images = [self.hdf['DataSet'][key]['TimePoint 0'][channel_keys[channel]]['Data'] for key in level_keys]
        # image_info = self.hdf.get('DataSetInfo')['Image'].attrs
        # print(eval(image_info['ExtMax3']))
        self.rois = []
        self.info = self.get_info()
        for i in self.info:
            # assure rois in order (z,y,x)
            self.rois.append(i['origin'] + i['data_shape'])
        self.roi = self.rois[0]


    def __getitem__(self, indices, level=0):
        z_min, z_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        x_min, x_max = indices[2].start, indices[2].stop
        z_slice = slice(z_min-self.rois[level][0],z_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        x_slice = slice(x_min-self.rois[level][2],x_max-self.rois[level][2])
        return self.images[level][z_slice,y_slice,x_slice]


    def from_roi(self, coords, level=0):
        # coords: [z_offset,y_offset,x_offset,z_size,y_size,x_size]
        # wanted coords
        z_min, z_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        x_min, x_max = coords[2], coords[5]+coords[2]

        # add padding
        # bounded coords
        [zlb,ylb,xlb] = self.rois[level][0:3] 
        [zhb,yhb,xhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)

        z_slice = slice(z_min-self.rois[level][0]+zlp,z_max-self.rois[level][0]-zhp) 
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        x_slice = slice(x_min-self.rois[level][2]+xlp,x_max-self.rois[level][2]-xhp)
        img = self.images[level][z_slice,y_slice,x_slice]

        padded = np.pad(img, ((zlp, zhp), (ylp, yhp), (xlp, xhp)), 'constant')

        return padded


    def from_slice(self,index,level,index_pos=0):
        """
        index_pos: 0 for z_slice, 1 for y_slice, 2 for x_slice
        """
        lb = self.rois[level][0:3] 
        hb = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        assert lb[index_pos] <= index < hb[index_pos], \
            f"Index {index} out of range for axis {index_pos}. Must be between {lb[index_pos]} and {hb[index_pos] - 1}."
            
        # Slicing the 3D array based on the given index and axis
        if index_pos == 0:  # Slice along the z-axis (extract a z_slice)
            slice_2d = self.images[level][index, lb[1]:hb[1], lb[2]:hb[2]]
        elif index_pos == 1:  # Slice along the y-axis (extract a y_slice)
            slice_2d = self.images[level][lb[0]:hb[0], index, lb[2]:hb[2]]
        elif index_pos == 2:  # Slice along the x-axis (extract an x_slice)
            slice_2d = self.images[level][lb[0]:hb[0], lb[1]:hb[1], index]
        else:
            raise ValueError(f"Invalid index_pos {index_pos}. Must be 0 (z), 1 (y), or 2 (x).")
        # Remove any extra dimensions
        slice_2d = np.squeeze(slice_2d)
        
        return slice_2d
        






    def from_local(self, coords, level=0):
        # coords: [z_offset,y_offset,x_offset,z_size,y_size,x_size]
        z_min, z_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        x_min, x_max = coords[2], coords[5]+coords[2]

        z_slice = slice(z_min,z_max) 
        y_slice = slice(y_min,y_max)
        x_slice = slice(x_min,x_max)
        return self.images[level][z_slice,y_slice,x_slice]


    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[0],data.shape[1],data.shape[2]],
                    'data_chunks':data.chunks,
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info



    def get_random_roi(self,
                    filter=lambda x:np.mean(x)>=150,
                    roi_size=(64,64,64),
                    level=0,
                    ):

        """
        random sample a roi of size (z_extend,y_extend,x_extend) 
        """
        foreground_sample_flag=False
        #shape: (z,y,x)
        info=self.get_info()
        shape=info[level]['data_shape']

        while not foreground_sample_flag:

            z_idx=np.random.randint(0,shape[0]-roi_size[0]) 
            y_idx=np.random.randint(0,shape[1]-roi_size[1]) 
            x_idx=np.random.randint(0,shape[2]-roi_size[2]) 
            roi=self.from_roi(coords=[z_idx,y_idx,x_idx,roi_size[0],roi_size[1],roi_size[2]],level=level)
            roi=roi.reshape(roi_size[0],roi_size[1],roi_size[2])
            roi=np.squeeze(roi)
            foreground_sample_flag=filter(roi)

        return roi, [z_idx,y_idx,x_idx]


