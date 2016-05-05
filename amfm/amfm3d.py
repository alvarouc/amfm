import numpy as np
import scipy.signal as ss
import scipy.ndimage.filters as sd
from amfm2d import genFilters

def chirp3d(N=100, max_freq=np.pi/2):

    grid = np.linspace(-max_freq,max_freq,N)
    x,y,z = np.meshgrid(grid, grid, grid)
    im = np.sin((x**2+y**2+z**2)*N/np.pi)
    return(im)


def applySeparableFilterBank(im,fb):

    out = []
    for fRow in fb:
        for fCol in fb:
            for ftime in fb:
                out.append(separable3dConv(im,fRow,fCol,ftime))

    return(out)

def separable3dConv(im,fRow,fCol,ftime):
    
    temp = sd.convolve1d(im,fRow,axis = 1)
    temp = sd.convolve1d(temp,fCol,axis=0)
    temp = sd.convolve1d(temp,ftime,axis=2)
    
    return(temp)

def qea(im):
    H = ss.hilbert(im,axis = 2)
    H = im+1j*H
    ia = np.abs(H)
    ip = np.angle(H)

    h1col = H[1:-1,:,:]
    h0col = H[:-2,:,:]
    h2col = H[2:,:,:]
    ifColSign = np.sign(np.real((h0col-h2col)/(2j*h1col)))
    ifCol = np.arccos((h2col+h0col)/(2*h1col))
    ifCol = (np.abs(ifCol)*ifColSign)/np.pi/2

    ifCol = np.pad(ifCol,((1,1),(0,0),(0,0)), mode='reflect')
    
    h0row = H[:,:-2,:]
    h1row = H[:,1:-1,:]
    h2row = H[:,2:,:]
    #ifxSign = np.sign(np.real((h2x-h0x)/(2j*h1x)))
    ifRow = np.arccos((h2row+h0row)/(2*h1row))
    ifRow = (np.abs(ifRow))/np.pi/2

    ifRow = np.pad(ifRow,((0,0),(1,1),(0,0)), mode='reflect')

    h0time = H[:,:,:-2]
    h1time = H[:,:,1:-1]
    h2time = H[:,:,2:]
    #ifxSign = np.sign(np.real((h2x-h0x)/(2j*h1x)))
    ifTime = np.arccos((h2time+h0time)/(2*h1time))
    ifTime = (np.abs(ifTime))/np.pi/2

    ifTime = np.pad(ifTime,((0,0),(0,0),(1,1)), mode='reflect')
    
    return(ia,ip,ifRow,ifCol,ifTime)

if __name__=="__main__":

    import imageio  
    im = chirp3d(100, np.pi/2)
    ia,ip,ifRow,ifCol,ifTime = qea(im)
    
    print(im.shape)
    print(ia.shape)
    print(ifRow.shape)
    print(ifCol.shape)
    print(ifTime.shape)

    temp = np.concatenate((ifRow,np.abs(ifCol),ifTime),axis=1)
    print(temp.shape)
    
    imageio.mimsave('chirp3d.gif', im)
    imageio.mimsave('chirp3d_ia.gif', ia)
    imageio.mimsave('chirp3d_ip.gif', ip)
    imageio.mimsave('chirp3d_if.gif', temp)
