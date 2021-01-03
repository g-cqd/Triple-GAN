( tr_images, tr_labels ), ( ts_images, ts_labels ) = mn.load_data()

dset_length = len( dset_images )

# Get first 100 labeled

## Seed the pseudo-random generator
seed( int ( round( time.time() * 1000 ) ) )

## Declare list variables to store
index_list = [None] * 100
dreq_images = [None] * 100
dreq_labels = [None] * 100

for i in range( 100 ):

    rand_int = int( random() * dset_length )
    while rand in index_list:
        rand_int = int( random() * dset_length )

    index_list[ i ] = rand_int

    dreq_images[ i ] = dset_images[ rand_int ]
    dreq_labels[ i ] = dset_labels[ rand_int ]

pt.figure( figsize=( 30, 30 ) )

for i in range( 100 ):
    pt.subplot( 10, 10, i + 1 )
    pt.title( f"Label : {dreq_labels[i]}" )
    pt.imshow( np.reshape( dreq_images[i], ( 28, 28 ) ) )

    