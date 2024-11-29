import os
import numpy as np
import mrcfile
from skimage.transform import resize



class DataPreprocessor:

    def __init__(self, config):
        self.config = config
        self.angles = None
        self.projections = None
        self.setup_directories()
        self.process_data()
        self.save_data()
    

    def setup_directories(self):
        '''Create the directories for saving the data.'''
        for dir_name in [self.config.path_save_data, self.config.path_save_data + "volumes"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)


    def downsample(self, projections, n1, n2):
        '''Downsample the projections.'''

        projections_downsample = np.zeros((projections.shape[0], n1, n2))

        for idx, proj in enumerate(projections):
            proj = resize(proj, (n1,n2), anti_aliasing=True)
            if self.config.transpose:
                projections_downsample[idx, :, :] = proj.T
            else:
                projections_downsample[idx, :, :] = proj

        return projections_downsample
    

    def crop(self, projections, n1, n2):
        '''Crop the projections. Keep the projection center.'''

        projections_crop = np.zeros((projections.shape[0], n1, n2))

        for i in range(projections.shape[0]):
            projections_crop[i,:,:] = projections[i, 
                                                  projections.shape[1]//2 - n1//2 : projections.shape[1]//2 + n1//2,
                                                  projections.shape[2]//2 - n2//2 : projections.shape[2]//2 + n2//2
                                                  ]
        return projections_crop
    
    def get_angles(self, angle_file):
        '''Generate the angles from the angle file or from a range. Prioritize the angle file'''

        if angle_file is None:
            angles = np.linspace(self.config.view_angle_min, self.config.view_angle_max, self.config.Nangles)
        else:
            angles = np.load(angle_file)
        
        return angles
    

    def save_data(self):
        np.save(self.config.path_save_data + "projections.npy", self.projections)
        np.save(self.config.path_save_data + "angles.npy", self.angles)
        np.savetxt(self.config.path_save_data + "angles.txt", self.angles)

        out = mrcfile.new(self.config.path_save_data + "projections.mrc", self.projections.astype(np.float32), overwrite=True)
        out.close()

    
    def process_data(self):
        '''Process the data and store it in angles and projections variables.'''
        n1, n2 = self.config.n1, self.config.n2

        path_volume = f"./data/{self.config.volume_name}/{self.config.volume_file}"
        
        mrc_stack = False

        if mrc_stack:
            projections = np.double(mrcfile.open(path_volume).data)
        else: # default
            mrc_files = []
            for proj_file in os.listdir(''):
                if proj_file.endswith(".mrc"):
                    mrc_files.append(np.double(mrcfile.open(proj_file).data))


        
        if self.config.downsample:
            print("Downsampling")
            projection_downsamples = self.downsample(projections, n1, n2)
        else:
            projection_downsamples = self.crop(projections, n1, n2)

        # TODO: Add denoising here
        if self.config.invert_projections:
            projection_downsamples = np.max(projection_downsamples) - projection_downsamples

        self.projections = projection_downsamples

        self.angles = self.get_angles(self.config.angle_file)



if __name__ == "__main__":
    dataprocess = DataPreprocessor()
    dataprocess.process_data()