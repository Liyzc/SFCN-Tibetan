from models.layers import *

class SFCN(nn.Module):
    def __init__(self, name, T, num_class=10, norm=None):
        super(SFCN, self).__init__()
        self.T = T
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.1307,), (0.3081,))
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        # self.neuron1 = LIFSpike()
        self.conv11 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=128,            
                kernel_size=3           
            ),  
            nn.BatchNorm2d(128),
        )
        self.ac11 = LIFSpike(T)
        self.conv12 = nn.Sequential(  
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
        )
        self.ac12 = LIFSpike(T)
        self.conv13 = nn.Sequential(  
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv21 = nn.Sequential(       
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
        )
        self.ac21 = LIFSpike(T)
        self.conv22 = nn.Sequential(  
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
        )
        self.ac22 = LIFSpike(T)
        self.conv23 = nn.Sequential(  
            nn.Conv2d(
                in_channels=128,
                out_channels=64, 
                kernel_size=3
            ),
            nn.BatchNorm2d(64),
        )
        self.ac23 = LIFSpike(T)
        self.conv24 = nn.Sequential(  
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifer = nn.Sequential(     
            
            nn.Conv2d(
                in_channels=64,
                out_channels=10, 
                kernel_size=3
            ),  
        )
    
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input):
        input = self.norm(input)
        # print(input.shape)
        if self.T > 0:
            input = add_dimention(input, self.T)
            input = self.merge(input)
            # print(input.shape)
        x = self.conv11(input)
        x = self.ac11(x)
        x = self.conv12(x)
        x = self.ac12(x)
        x = self.conv13(x)
        # print(x.shape)
        x = self.conv21(x)
        x = self.ac21(x)
        x = self.conv22(x)
        x = self.ac22(x)
        x = self.conv23(x)
        x = self.ac23(x)
        x = self.conv24(x)
        # x = F.avg_pool2d(x, 4)
        # print(x.shape)
        # x = torch.flatten(x, 2)
        x = self.classifer(x)
        if self.T > 0:
            x = self.expand(x).squeeze()
        else:
            x = x.squeeze()
        # x = x.mean(0)
        # print(x.shape)
        
        return x#.mean(0).max(dim = 1)
    
class AFCN(nn.Module):
    def __init__(self, name, T, num_class=10, norm=None):
        super(AFCN, self).__init__()
        self.T = T
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.1307,), (0.3081,))
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=128,            
                kernel_size=3           
            ),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(       
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=3
            ),     
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=64, 
                kernel_size=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifer = nn.Sequential(     
            
            nn.Conv2d(
                in_channels=64,
                out_channels=10, 
                kernel_size=3
            ),  
        )
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.classifer(x)
        # print(x.size())
        return x.squeeze()
        
        # return x#.mean(0).max(dim = 1)