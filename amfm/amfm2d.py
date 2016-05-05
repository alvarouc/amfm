import numpy as np
import scipy.signal as ss
import scipy.ndimage.filters as sd
import matplotlib.pyplot as plt


def chirp2d(N=500, max_freq=np.pi/2):

    grid = np.linspace(-max_freq,max_freq,N)
    x,y = np.meshgrid(grid, grid)
    im = np.sin((x**2+y**2)*N/np.pi)
    return(im)

def genFilters():

    bp = [ ]
    bp.append(ss.remez(50, [0, 0.02, 0.05, 0.5], [1,0]))
    bp.append(ss.remez(50, [0, 0.02, 0.05, 0.20, 0.25, 0.5], [0, 1, 0]))
    bp.append(ss.remez(50, [0, 0.20, 0.25, 0.5], [0, 1],
                       type = "hilbert"))

    return bp

def plotFilters(bp):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for bpass in bp:
        freq, response = ss.freqz(bpass)
        ampl = np.abs(response)
        
        ax1.semilogy(freq/(2*np.pi), ampl, '-')  # freq in Hz
        ax1.set_ylim(1e-4,1.1)

    plt.show()

def separable2dConv(im,fRow,fCol):
    # run filter through rows
    temp = sd.convolve1d(im,fRow,axis = 1)
    # run filter through columns
    temp = sd.convolve1d(temp,fCol,axis=0)
    return(temp)
    
def applySeparableFilterBank(im,fb):

    out = []
    for fRow in fb:
        for fCol in fb:
            out.append(separable2dConv(im,fRow,fCol))

    return(out)

def qea(im):
    H = ss.hilbert(im,axis = 1)
    H = im+1j*H
    ia = np.abs(H)
    ip = np.angle(H)

    h1col = H[1:-1,:]
    h0col = H[:-2,:]
    h2col = H[2:,:]
    ifColSign = np.sign(np.real((h0col-h2col)/(2j*h1col)))
    ifCol = np.arccos((h2col+h0col)/(2*h1col))
    ifCol = (np.abs(ifCol)*ifColSign)[:,1:-1]/np.pi/2

    h0row = H[:,:-2]
    h1row = H[:,1:-1]
    h2row = H[:,2:]
    #ifxSign = np.sign(np.real((h2x-h0x)/(2j*h1x)))
    ifRow = np.arccos((h2row+h0row)/(2*h1row))
    ifRow = (np.abs(ifRow))[1:-1,:]/np.pi/2

    return(ia,ip,ifRow,ifCol)

def amfm(im):

    im = np.array(im,dtype=np.float)
    bp = genFilters()
    imbp= applySeparableFilterBank(im,bp)

    out = list(map(qea, imbp))

    ias= np.array([ia for ia,ip,ifRow,ifCol in out])
    ifRows = np.array([ifRow for ia,ip,ifRow,ifCol in out])
    ifCols = np.array([ifCol for ia,ip,ifRow,ifCol in out])
    ips = np.array([ip for ia,ip,ifRow,ifCol in out])

    idx = ias.argmax(axis=0)
    idx = idx[1:-1,1:-1]

    # Applying DCA
    temp = np.zeros(idx.shape)
    for i in range(len(imbp)):
        temp[idx==i]=ifCols[i,...][idx==i]
    ifCol = np.copy(temp)

    temp = np.zeros(idx.shape)
    for i in range(len(imbp)):
        temp[idx==i]=ifRows[i,...][idx==i]
    ifRow = np.copy(temp)

    temp = np.zeros(idx.shape)
    for i in range(len(imbp)):
        temp[idx==i]=ips[i,...][idx==i]
    ip = np.copy(temp)

    ia = ias.max(axis=0)

    mypad = lambda x: np.pad(x,((1,1),(1,1)),mode = "reflect")
    ifRow = mypad(ifRow)
    ifCol = mypad(ifCol)
    return(ia,ip,ifRow,ifCol)


def plotAMFM(im, ifRow,ifCol):

    plt.imshow(im,interpolation="nearest",
               cmap = "Greys_r")
    Row,Col = np.meshgrid(range(0,im.shape[0]),
                          range(0,im.shape[1]))
    sp= np.round(max(im.shape)/30)
    plt.quiver(Col[::sp,::sp],Row[::sp,::sp],
               ifCol[::sp,::sp],ifRow[::sp,::sp],
               scale = 10, color = "r",
               pivot = "mid", width = 0.004)
    plt.title("Instantaneous frequency") 
    
    
if __name__=="__main__":

    im = chirp2d(500)
    #tm = Image.open("/home/aulloa/Pictures/zebra.jpg")
    #im = np.array(list(tm.getdata())).reshape(tm.size[1],
    #                                          tm.size[0],
    #                                          3)
    #im = im[:1000,:1000,0]
    
    ia,ip,ifRow,ifCol= amfm(np.array(im,dtype=np.float))
    
    
    plt.figure()
    #plt.subplot(1,3,1)
    plotAMFM(im,ifRow,ifCol)
    plt.title("Instantaneous frequency")
    
    #plt.subplot(1,3,2)
    #plt.imshow(ia, interpolation="nearest",
    #           cmap = "Greys_r")
    #plt.title("Instantaneous Amplitude")

    #plt.subplot(1,3,3)
    #plt.imshow(ip, interpolation="nearest",
    #           cmap = "Greys_r")
    #plt.title("Instantaneous Phase")

    plt.show()

#plt.figure()
#plt.imshow(im,cmap = "Greys",#np.abs(np.fft.fft2(im)),
#           interpolation ="nearest");
#plt.figure()
#plt.imshow(np.abs(np.fft.fft2(im)),
#           cmap = "Greys",
#           interpolation ="nearest");
#plt.show()
