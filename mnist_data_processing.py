# PROVIDE YOUR DOWNLOAD DIRECTORY HERE
#Sample path '/home/pune1/pranav/Deeplearning/data/mnist/data/'

datapath =  ''

files = os.listdir(datapath)


def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)


data_dict = {}
for file in files:
    print('Reading ',file)
    with open (datapath+file,'rb') as f:
        data = f.read()
        type = get_int(data[:4])   # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
        length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
        if (type == 2051):
            category = 'images'
            num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
            num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
            parsed = numpy.frombuffer(data,dtype = numpy.uint8, offset = 16)  # READ THE PIXEL VALUES AS INTEGERS
            parsed = parsed.reshape(length,num_rows,num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]           
        elif(type == 2049):
            category = 'labels'
            parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8) # READ THE LABEL VALUES AS INTEGERS
            parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]                           
        if (length==10000):
            set = 'test'
        elif (length==60000):
            set = 'train'
        data_dict[set+'_'+category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY


#for storing images and keeping the dictionary containing images and labels
datapath = '' # PATH WHERE IMAGES WILL BE SAVED
sets = ['train','test']
for set in sets:   # FOR TRAIN AND TEST SET
    images = data_dict[set+'_images']   # IMAGES
    labels = data_dict[set+'_labels']   # LABELS
    no_of_samples = images.shape[0]     # NUBMER OF SAMPLES
    for indx in range (no_of_samples):  # FOR EVERY SAMPLE
        print(set, indx)
        image = images[indx]            # GET IMAGE
        label = labels[indx]            # GET LABEL
        if not os.path.exists(datapath+set+'/'+str(label)+'/'):    # IF DIRECTORIES DO NOT EXIST THEN 
            os.makedirs (datapath+set+'/'+str(label)+'/')       # CREATE TRAIN/TEST DIRECTORY AND CLASS SPECIFIC SUBDIRECTORY
        filenumber = len(os.listdir(datapath+set+'/'+str(label)+'/'))  # NUMBER OF FILES IN THE DIRECTORY FOR NAMING THE FILE
        imsave(datapath+set+'/'+str(label)+'/%05d.png'%(filenumber),image)  # SAVE THE IMAGE WITH PROPER NAME


#keeping a dictionary having all the data, to be used later
import pickle
datapath = ''


# DUMPING THE DICTIONARY INTO A PICKLE 
with open(datapath+'MNISTData.pkl', 'wb') as fp :
    pickle.dump(data_dict, fp)

    